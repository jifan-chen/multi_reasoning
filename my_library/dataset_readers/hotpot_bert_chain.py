import json
import logging
import re
from rouge import Rouge
from typing import Dict, List, Tuple, Any
from collections import Counter
from overrides import overrides
from allennlp.data.fields import Field, TextField, IndexField, ArrayField, SpanField, \
    MetadataField, LabelField, ListField, AdjacencyField
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
rouge = Rouge()


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        question_passage_tokens: List[Token],
                                        question_passage_offsets: List[Tuple[int, int]],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        token_spans: List[Tuple[int, int]] = None,
                                        token_spans_sp: List[Tuple[int, int]] = None,
                                        token_spans_sent: List[Tuple[int, int]] = None,
                                        sent_labels: List[int] = None,
                                        answer_texts: List[str] = None,
                                        additional_metadata: Dict[str, Any] = None,
                                        para_limit: int = 2250) -> Instance:
    """
    Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
    in a reading comprehension model.
    Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
    ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
    and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
    fields, which are both ``IndexFields``.
    Parameters
    ----------
    question_tokens : ``List[Token]``
        An already-tokenized question.
    question_passage_tokens : ``List[Token]``
        An already-tokenized passage that contains the answer to the given question.
    token_indexers : ``Dict[str, TokenIndexer]``
        Determines how the question and passage ``TextFields`` will be converted into tensors that
        get input to a model.  See :class:`TokenIndexer`.
    passage_text : ``str``
        The original passage text.  We need this so that we can recover the actual span from the
        original passage that the model predicts as the answer to the question.  This is used in
        official evaluation scripts.
    token_spans : ``List[Tuple[int, int]]``, optional
        Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
        a list because there might be several possible correct answer spans in the passage.
        Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
        annotations on the dev set; this will select the span that the most annotators gave as
        correct).
    answer_texts : ``List[str]``, optional
        All valid answer strings for the given question.  In SQuAD, e.g., the training set has
        exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
        possible answers, which are the aliases for the known correct entity.  This is put into the
        metadata for use with official evaluation scripts, but not used anywhere else.
    passage_dep_heads : ``List[int]``, optional
        The dependency parents for each token in the passage, zero-indexing.
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
        you want any other metadata to be associated with each instance, you can pass that in here.
        This dictionary will get added to the ``metadata`` dictionary we already construct.
    para_limit : ``int``, indicates the maximum length of a given article
    """
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}

    limit = len(question_passage_tokens) if para_limit > len(question_passage_tokens) else para_limit
    question_passage_tokens = question_passage_tokens[:limit]

    # This is separate so we can reference it later with a known type.
    question_passage_field = TextField(question_passage_tokens, token_indexers)

    fields['question_passage'] = question_passage_field
    # print(question_passage_field)
    # print(question_tokens)
    # input()
    sent_labels_: List[Field] = []
    if token_spans_sent:
        for label in sent_labels:
            sent_labels_.append(LabelField(label, skip_indexing=True))

    fields['sent_labels'] = ListField(sent_labels_)

    # filter spans that exceed para limit so that the info in metadata is correct
    token_spans_sent = [(s, e if e < limit else limit - 1) for s, e in token_spans_sent if s < limit]
    token_spans_sp = [(s, e if e < limit else limit - 1) for s, e in token_spans_sp if s < limit]
    sent_labels = sent_labels[:len(token_spans_sent)]

    metadata = {'original_passage': passage_text, 'token_offsets': question_passage_offsets,
                'question_tokens': [token.text for token in question_tokens],
                'passage_tokens': [token.text for token in question_passage_tokens],
                'token_spans_sp': token_spans_sp,
                'token_spans_sent': token_spans_sent,
                'sent_labels': sent_labels}
    if answer_texts:
        metadata['answer_texts'] = answer_texts

    # print('answer:', answer_texts[0])
    # print('answer_text:', answer_texts)
    if answer_texts[0] == 'yes':
        fields['q_type'] = LabelField(1, skip_indexing=True)
        fields['span_start'] = IndexField(-100, question_passage_field)
        fields['span_end'] = IndexField(-100, question_passage_field)
    elif answer_texts[0] == 'no':
        fields['q_type'] = LabelField(2, skip_indexing=True)
        fields['span_start'] = IndexField(-100, question_passage_field)
        fields['span_end'] = IndexField(-100, question_passage_field)
    else:
        fields['q_type'] = LabelField(0, skip_indexing=True)

        if token_spans:

            candidate_answers: Counter = Counter()
            for span_start, span_end in token_spans:
                candidate_answers[(span_start, span_end)] += 1
            for s, e in candidate_answers:
                if not any([sp_s <= s and e <= sp_e for sp_s, sp_e in token_spans_sp]):
                    candidate_answers[(s, e)] = 0
            span_start, span_end = candidate_answers.most_common(1)[0][0]

            if span_start > para_limit or span_end > para_limit:
                # print('span_start, span_end:', span_start, span_end)
                fields['span_start'] = IndexField(-100, question_passage_field)
                fields['span_end'] = IndexField(-100, question_passage_field)
            else:
                fields['span_start'] = IndexField(span_start, question_passage_field)
                fields['span_end'] = IndexField(span_end, question_passage_field)
        else:
            fields['span_start'] = IndexField(-100, question_passage_field)
            fields['span_end'] = IndexField(-100, question_passage_field)

    # print('start:', fields['span_start'])
    # print('end:', fields['span_end'])
    # print(metadata['passage_tokens'][span_start:span_end + 1])
    # print(question_passage_tokens)
    # print(answer_texts)
    # input()

    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)


@DatasetReader.register("hotpot_bert_chain")
class HotpotDatasetReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 para_limit: int = 2250,
                 sent_limit: int = 70,
                 rerank_by_sp: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._para_limit = para_limit
        self._sent_limit = sent_limit
        self._rerank_by_sp = rerank_by_sp
        self.count = 0
        self._answer_count = 0.0
        self._avg_chain_length = 0

    @staticmethod
    def find_all_span_starts(answer, context):
        return [m.start() for m in re.finditer(re.escape(answer), context)]

    @staticmethod
    def find_span_starts(span, context):
        return re.search(re.escape(span), context).start()

    @staticmethod
    def get_all_dep_pairs(heads):
        pass

    @staticmethod
    def f1_score(pred, gold):

        TP = [p in gold for p in pred].count(True)
        TP_FP = len(pred)
        TP_FN = len(gold)
        # print('pred:', pred)
        # print('gold:', gold)
        # print('TP:', TP)
        # print('TP_FP:', TP_FP)
        # print('TP_FN:', TP_FN)
        precision = TP / (TP_FP + 1e-10)
        recall = TP / (TP_FN + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1

    def chain_rerank(self, pred_chains, sp_set):
        reranked_chains = []
        if len(sp_set) > 0:
            for chain in pred_chains:
                reranked_chains.append([chain, self.f1_score(chain, sp_set)])
            reranked_chains = sorted(reranked_chains, key=lambda rc: rc[1], reverse=True)
            reranked_chains = [chain for chain, score in reranked_chains]
            return reranked_chains
        else:
            return pred_chains

    @staticmethod
    def f1_score_by_question(chain, question_text):
        if len(chain) > 10 and len(question_text) > 10:
            # print(chain)
            # print(question_text)
            rouge_score = rouge.get_scores(chain, question_text)
            return rouge_score[0]['rouge-1']['f']
        else:
            return 0.0

    def chain_rerank_by_question(self, pred_chains, question_text, answer_text, all_sents):
        reranked_chains = []
        for chain in pred_chains:
            f1_avg = 0.0
            answer_count = 0.0
            for e in chain:
                try:
                    chain_text = all_sents[e]
                except IndexError:
                    continue
                f1_avg += self.f1_score_by_question(chain_text, question_text)
                if answer_text in chain_text:
                    answer_count += 1
            f1_avg = f1_avg / (len(chain) + 1e-20)
            f1_avg += answer_count
            reranked_chains.append([chain, f1_avg])
        reranked_chains = sorted(reranked_chains, key=lambda rc: rc[1], reverse=True)

        reranked_chains = [chain for chain, score in reranked_chains]
        return reranked_chains

    @staticmethod
    def preprocess_global_info(pragraphs, sp_set):
        global_id = 0
        all_sents = []
        sp_facts_id = []
        sent_labels = []
        for para in pragraphs:
            cur_title, cur_para = para[0], para[1]
            for sent_id, sent in enumerate(cur_para):
                all_sents.append(sent)
                if (cur_title, sent_id) in sp_set:
                    sp_facts_id.append(global_id)
                    sent_labels.append(1)
                else:
                    sent_labels.append(0)
                global_id += 1
        return all_sents, sp_facts_id, sent_labels

    @staticmethod
    def get_topK_chains(pred_chains, top_k, total_num_chains):
        combined_chains_id = []
        for chain in pred_chains[:top_k]:
            for e in chain:
                if e not in combined_chains_id and len(combined_chains_id) < total_num_chains:
                    combined_chains_id.append(e)
        return combined_chains_id

    def process_raw_instance(self, article):
        article_id = article['_id']
        paragraphs = article['context']
        combined_chains = []
        concat_qp = ""
        question_passage_tokens = []
        question_passage_offsets = []
        sent_starts = []
        sent_ends = []
        answer_in = []
        question_text = article['question'].strip().replace("\n", "")
        answer_text = article['answer'].strip().replace("\n", "")
        sp_set = set(list(map(tuple, article['supporting_facts'])))
        all_sents, sp_facts_id, sent_labels = self.preprocess_global_info(paragraphs, sp_set)
        pred_chains_sp = self.chain_rerank(article['pred_chains'], sp_facts_id) if self._rerank_by_sp \
            else article['pred_chains']
        pred_chains_ques = self.chain_rerank_by_question(article['pred_chains'], question_text, answer_text, all_sents) if self._rerank_by_sp \
            else article['pred_chains']
        combined_chains_id = self.get_topK_chains(pred_chains_ques, top_k=1, total_num_chains=8)
        if len(combined_chains_id) == 0:
            combined_chains_id = self.get_topK_chains(pred_chains_ques, top_k=2, total_num_chains=7)
        # combined_chains_id_sp = self.get_topK_chains(pred_chains_sp, top_k=2, total_num_chains=5)
        for id in combined_chains_id:
            if answer_text in all_sents[id]:
                self._answer_count += 1
                self._avg_chain_length += len(combined_chains_id)
                break
        # print('by question:', combined_chains_id)
        # print('by sp:', combined_chains_id_sp)
        # print('sp set:', sp_facts_id)
        # input()
        tokenized_ques = self._tokenizer.tokenize(question_text)
        tokenized_ques = [Token(text=tk.text, idx=tk.idx + 1) for tk in tokenized_ques]
        tokenized_ques.insert(0, Token(text='[CLS]', idx=0))
        tokenized_ques.append(Token(text='[SEP]', idx=tokenized_ques[-1].idx + 1))
        appended_question_text = "[CLS]{}[SEP]".format(question_text)
        sent_offset = [(tk.idx + len(concat_qp),
                        tk.idx + len(tk.text) + len(concat_qp)) for tk in tokenized_ques]
        question_passage_offsets.extend(sent_offset)
        concat_qp += appended_question_text
        question_passage_tokens.extend(tokenized_ques)

        avg_sent_len = 0.0

        sent_lengths = [len(all_sents[sent_id]) < self._sent_limit for sent_id in combined_chains]
        limit = self._sent_limit
        if 1 not in sent_lengths:
            limit = 1000

        for sent_id in combined_chains_id:
            sent = all_sents[sent_id]
            tokenized_sent = self._tokenizer.tokenize(sent)
            if len(tokenized_sent) < limit:
                combined_chains.append(sent)
                tokenized_sent = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_sent]
                avg_sent_len += len(tokenized_sent)
                sent_offset_qp = [(tk.idx + len(concat_qp),
                                   tk.idx + len(tk.text) + len(concat_qp)) for tk in tokenized_sent]

                if sent_offset:
                    sent_start = sent_offset[0][0]
                    sent_end = sent_offset[-1][1]
                    sent_starts.append(sent_start)
                    sent_ends.append(sent_end)

                question_passage_offsets.extend(sent_offset_qp)
                concat_qp += sent
                question_passage_tokens.extend(tokenized_sent)

        concat_qp += '[SEP]'
        question_passage_tokens.append(Token(text='[SEP]', idx=question_passage_tokens[-1].idx + 1))
        question_passage_offsets.append((len(concat_qp), len(concat_qp) + len('[SEP]')))
        span_starts = self.find_all_span_starts(answer_text, concat_qp)
        span_ends = [start + len(answer_text) for start in span_starts]
        sp_starts = [self.find_span_starts(s, concat_qp) for s in combined_chains]
        sp_ends = [start + len(span) for span, start in zip(combined_chains, sp_starts)]

        self.count += avg_sent_len / len(sp_facts_id)
        return (question_text,
                concat_qp,
                zip(span_starts, span_ends),
                zip(sp_starts, sp_ends),
                zip(sent_starts, sent_ends),
                sent_labels,
                [answer_text],
                question_passage_tokens,
                question_passage_offsets,
                article_id)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")

        for article in dataset:
            processed_article = self.process_raw_instance(article)
            instance = self.text_to_instance(*processed_article)
            yield instance
        print(self._answer_count / len(dataset))
        print(self._avg_chain_length / len(dataset))

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         char_spans_sp: List[Tuple[int, int]] = None,
                         char_spans_sent: List[Tuple[int, int]] = None,
                         sent_labels: List[int] = None,
                         answer_texts: List[str] = None,
                         question_passage_tokens: List[Token] = None,
                         question_passage_offsets: List[Tuple[int, int]] = None,
                         article_id: str = None) -> Instance:

        char_spans = char_spans or []
        char_spans_sp = char_spans_sp or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        token_spans_sp: List[Tuple[int, int]] = []
        token_spans_sent: List[Tuple[int, int]] = []

        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(question_passage_offsets,
                                                                         (char_span_start, char_span_end))
            # print(span_start, span_end)
            # print(question_passage_tokens[span_start:span_end+1])
            # print(answer_texts)
            # print(passage_text)

            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", question_passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", question_passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        for char_span_sp_start, char_span_sp_end in char_spans_sp:
            (span_start_sp, span_end_sp), error = util.char_span_to_token_span(question_passage_offsets,
                                                                               (char_span_sp_start, char_span_sp_end))
            token_spans_sp.append((span_start_sp, span_end_sp))

        for char_span_sent_start, char_span_sent_end in char_spans_sent:
            (span_start_sent, span_end_sent), error = util.char_span_to_token_span(question_passage_offsets,
                                                                                   (char_span_sent_start,
                                                                                    char_span_sent_end))
            token_spans_sent.append((span_start_sent, span_end_sent))
        # input()
        tokenized_ques = self._tokenizer.tokenize(question_text)
        tokenized_ques = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_ques]

        return make_reading_comprehension_instance(tokenized_ques,
                                                   question_passage_tokens,
                                                   question_passage_offsets,
                                                   self._token_indexers,
                                                   passage_text,
                                                   token_spans,
                                                   token_spans_sp,
                                                   token_spans_sent,
                                                   sent_labels,
                                                   answer_texts,
                                                   additional_metadata={'_id': article_id},
                                                   para_limit=self._para_limit)


if __name__ == '__main__':
    reader = HotpotDatasetReader()
    reader.read("/backup2/jfchen/data/hotpot/hotpot_test.json")