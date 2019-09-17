import json
import logging
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from collections import Counter
from overrides import overrides
from allennlp.data.fields import Field, TextField, IndexField, ArrayField, SpanField, \
    MetadataField, LabelField, ListField
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
wordpiece_tokenizer = bert_tokenizer.wordpiece_tokenizer.tokenize


def process_sent_spans(para_sent_token_spans, passage_para_field):
    passage_para_spans: List[ListField[Field]] = []
    if para_sent_token_spans:
        for para_spans, para_field in zip(para_sent_token_spans, passage_para_field):
            para_spans_field = []
            for start, end in para_spans:
                para_spans_field.append(SpanField(start, end, para_field))
            passage_para_spans.append(ListField(para_spans_field))
    return passage_para_spans


def process_evidence_chains(evd_possible_chains, sent_labels_, fields):
    evd_possible_chains_ = []
    if evd_possible_chains is not None:
        for chain in evd_possible_chains:
            if len(chain) == 0 or any([s_idx >= len(sent_labels_) for s_idx in chain]):
                # if there is no possible chain or any selected sentence in the chain exceeds para_limit,
                # ignore the instance.
                # the chain start with 0 will be filtered out in RLBidirectionalAttentionFlow Module.
                chain = [0]
            else:
                # Since indice 0 is for eos, shifts by one
                # Also add eos at the end
                chain = [s_idx + 1 for s_idx in chain] + [0]
            evd_possible_chains_.append(chain)
        fields['evd_chain_labels'] = ListField([ArrayField(np.array(ch), padding_value=0)
                                                for ch in evd_possible_chains_])
        #print('oracle:', fields['evd_chain_labels'])
    return evd_possible_chains_


def process_answer_spans(token_spans, token_spans_sp, answer_texts, passage_field, para_limit, fields):
    # if yes/no question, set span_start, span_end to IGNORE_INDEX
    if answer_texts[0] == 'yes':
        fields['q_type'] = LabelField(1, skip_indexing=True)
        fields['span_start'] = IndexField(-100, passage_field)
        fields['span_end'] = IndexField(-100, passage_field)
    elif answer_texts[0] == 'no':
        fields['q_type'] = LabelField(2, skip_indexing=True)
        fields['span_start'] = IndexField(-100, passage_field)
        fields['span_end'] = IndexField(-100, passage_field)
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

            if span_start >= para_limit or span_end >= para_limit:
                fields['span_start'] = IndexField(-100, passage_field)
                fields['span_end'] = IndexField(-100, passage_field)
            else:
                fields['span_start'] = IndexField(span_start, passage_field)
                fields['span_end'] = IndexField(span_end, passage_field)
        else:
            fields['span_start'] = IndexField(-100, passage_field)
            fields['span_end'] = IndexField(-100, passage_field)


def make_meta_data(passage_text, question_tokens, passage_para_tokens,
                   sent_labels, answer_texts, evd_possible_chains, evd_possible_chains_, ans_sent_idxs, article_id):
    # 0 denotes eos, shifts by one
    if ans_sent_idxs is not None:
        ans_sent_idxs = [s_idx + 1 for s_idx in ans_sent_idxs if s_idx < len(sent_labels)]
    metadata = {'original_passage': passage_text,
                'question_tokens': [token.text for token in question_tokens],
                'passage_sent_tokens': [[token.text for token in sent_tokens] for sent_tokens in passage_para_tokens],
                'sent_labels': sent_labels,
                '_id': article_id}
    if answer_texts:
        metadata['answer_texts'] = answer_texts
    if evd_possible_chains is not None:
        metadata['evd_possible_chains'] = evd_possible_chains_
    if ans_sent_idxs is not None:
        metadata['ans_sent_idxs'] = ans_sent_idxs
    return metadata


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_sent_tokens: List[List[Token]],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        sent_labels: List[int] = None,
                                        answer_texts: List[str] = None,
                                        evd_possible_chains: List[List[int]] = None,
                                        ans_sent_idxs: List[int] = None,
                                        sents_token_span: List[Tuple[int, int]] = None,
                                        article_id: str = None,
                                        para_limit: int = 2250) -> Instance:
    """
    Parameters
    ----------
    question_tokens : ``List[Token]``
        An already-tokenized question.
    passage_tokens : ``List[Token]``
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
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
        you want any other metadata to be associated with each instance, you can pass that in here.
        This dictionary will get added to the ``metadata`` dictionary we already construct.
    para_limit : ``int``, indicates the maximum length of a given article
    """
    fields: Dict[str, Field] = {}
    # limit = len(passage_tokens) if para_limit > len(passage_tokens) else para_limit
    # passage_tokens = passage_tokens[:limit]
    # This is separate so we can reference it later with a known type.
    passage_sent_field = []
    sents_span_field = []
    for sent_tokens, sent_token_span in zip(passage_sent_tokens, sents_token_span):
        sent_start, sent_end = sent_token_span[0], sent_token_span[1]
        sent_field = TextField(sent_tokens, token_indexers)
        sents_span_field.append(SpanField(sent_start, sent_end, sent_field))
        passage_sent_field.append(sent_field)
    # sent_spans: list of [SpanFiled[sent_start, sent_end]], denote the start and end offset for each sentence
    # sent_labels_: list of [label], denote the whether a sentence is a supporting fact
    # print(sent_spans)
    # print(passage_para_field)
    # input()
    fields['sent_labels'] = ListField([LabelField(label, skip_indexing=True) for label in sent_labels])
    fields['passage'] = ListField(passage_sent_field)
    fields['sentence_spans'] = ListField(sents_span_field)
    # print(fields['passage'])

    fields['question'] = TextField(question_tokens, token_indexers)
    #print('question:', fields['question'])
    #print('sent_labels:', fields['sent_labels'])
    # filter spans that exceed para limit so that the info in metadata is correct
    # token_spans_sent = [(s, e if e < limit else limit - 1) for s, e in token_spans_sent if s < limit]
    # token_spans_sp = [(s, e if e < limit else limit - 1) for s, e in token_spans_sp if s < limit]
    # process_answer_spans(token_spans, token_spans_sp, answer_texts, passage_field, para_limit, fields)
    evd_possible_chains_ = process_evidence_chains(evd_possible_chains, sent_labels, fields)
    metadata = make_meta_data(passage_text, question_tokens, passage_sent_tokens,
                              sent_labels, answer_texts, evd_possible_chains,
                              evd_possible_chains_, ans_sent_idxs, article_id)
    fields['metadata'] = MetadataField(metadata)
    #input()
    return Instance(fields)


@DatasetReader.register("hotpot_reader_bert_sentence")
class HotpotDatasetReader(DatasetReader):
    """
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    para_limit : The max length of the input document.
    sent_limit : The max length of a single sentence in the document.
    filter_compare_q : Filter the yes/no questions.
    chain : read from the rule based chains or read from predicted chains(for fine-tune)
    lazy : lazy reading
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 para_limit: int = 2250,
                 sent_limit: int = 75,
                 word_piece_limit: int = 142,
                 context_limit: int = 20,
                 training: bool = False,
                 filter_compare_q: bool = False,
                 chain: str = 'rb',
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._para_limit = para_limit
        self._sent_limit = sent_limit
        self._context_limit = context_limit
        self._word_piece_limit = word_piece_limit
        self._filter_compare_q = filter_compare_q
        self.chain = chain
        self.training = training

    @staticmethod
    def find_all_span_starts(answer, context):
        return [m.start() for m in re.finditer(re.escape(answer), context)]

    @staticmethod
    def find_span_starts(span, context):
        return re.search(re.escape(span), context).start()

    @staticmethod
    def get_all_dep_pairs(heads):
        pass

    def process_raw_instance(self, article):
        article_id = article['_id']
        paragraphs = article['context']
        evd_possible_chains = []
        if self.chain == 'rb':
            evd_possible_chains = article.get("possible_chain", None)
            evd_possible_chains = [evd_possible_chains] if evd_possible_chains is not None else None
        elif self.chain == 'nn':
            evd_possible_chains = article.get("pred_chains", None)

        question_text = article['question'].strip().replace("\n", "")
        answer_text = article['answer'].strip().replace("\n", "")
        sp_set = set(list(map(tuple, article['supporting_facts'])))
        concat_article = ""
        passage_sent_tokens = []
        supporting_facts = []
        # passage_offsets, used to convert char-span to token-span
        para_sent_spans = []
        # labels denoting whether a sentence is a supporting fact or not
        sent_labels = []
        # labels denoting whether a sentence contains the answer or not
        ans_sent_idxs = []
        sents_offset = []
        sent_starts = []
        sent_ends = []
        tokenized_ques = self._tokenizer.tokenize(question_text)
        # magic number 6 -> offset for the special token [CLS]
        tokenized_stand_alone_ques = [Token(text=tk.text, idx=tk.idx + 6) for tk in tokenized_ques]
        tokenized_stand_alone_ques.insert(0, Token(text='[CLS]', idx=0))
        tokenized_stand_alone_ques.append(Token(text='[SEP]', idx=tokenized_stand_alone_ques[-1].idx + 1))
        appended_question_text = "[CLS]{}[SEP]".format(question_text)

        # pre compute all the tokenized sent rep
        tokenized_sents = []
        for para in paragraphs:
            cur_title, cur_para = para[0], para[1]
            tokenized_sents_per_para = []
            for sent_id, sent in enumerate(cur_para):
                tokenized_sent = self._tokenizer.tokenize(sent)
                if len(tokenized_sent) == 0:
                    tokenized_sent.append(Token(text='[EMPTY]', idx=0))
                tokenized_sents_per_para.append(tokenized_sent)
            tokenized_sents.append(tokenized_sents_per_para)

        for para_id, para in enumerate(paragraphs):
            cur_title, cur_para = para[0], para[1]

            for sent_id, sent in enumerate(cur_para):
                # Tokenize each sentence
                sent_offset = []
                tokenized_sent = tokenized_sents[para_id][sent_id]

                if len(tokenized_sent) > 0:
                    tokenized_concat_ques = [Token(text=tk.text, idx=tk.idx + 6) for tk in tokenized_ques]
                    tokenized_concat_ques.insert(0, Token(text='[CLS]', idx=0))
                    tokenized_concat_ques.append(Token(text='[SEP]', idx=tokenized_concat_ques[-1].idx + 1))
                    appended_concat_question_text = "[CLS]{}[SEP]".format(question_text)

                    sent_offset = [(tk.idx, tk.idx + len(tk.text)) for tk in tokenized_concat_ques]
                    base_sent_offset = len(appended_concat_question_text)

                    # if the first sentence, pad the next sentence
                    if sent_id == 0:
                        # convert to Allen's Token for parallel reading
                        # assert sent_id + 1 < len(cur_para)
                        tokenized_sent = [Token(text=tk.text, idx=len(appended_concat_question_text) + tk.idx)
                                          for tk in tokenized_sent][:self._sent_limit]
                        tokenized_sent.append(Token(text='[SEP]', idx=tokenized_sent[-1].idx + 1))
                        appended_sent_text = "{}[SEP]".format(sent)

                        if sent_id + 1 < len(cur_para):

                            next_tokenized_sent = tokenized_sents[para_id][sent_id+1]
                            next_tokenized_sent = [Token(text=tk.text,
                                                   idx=len(appended_concat_question_text) + len(appended_sent_text) +
                                                         tk.idx)
                                                   for tk in next_tokenized_sent][:self._sent_limit]
                            next_tokenized_sent.append(Token(text='[SEP]', idx=next_tokenized_sent[-1].idx + 1))

                        else:
                            next_tokenized_sent = []
                        # print('first sentence:', tokenized_sent)
                        # print("next sentence:", next_tokenized_sent)
                        tokenized_concat_ques.extend(tokenized_sent)
                        tokenized_concat_ques.extend(next_tokenized_sent)
                        sent_offset.extend([(tk.idx, tk.idx + len(tk.text)) for tk in tokenized_sent])
                        sent_offset.extend([(tk.idx, tk.idx + len(tk.text)) for tk in next_tokenized_sent])
                        sent_start = tokenized_sent[0].idx
                        sent_end = tokenized_sent[-1].idx

                    elif sent_id == len(cur_para) - 1:
                        # assert sent_id - 1 >= 0
                        if sent_id - 1 >= 0:
                            prev_tokenized_sent = tokenized_sents[para_id][sent_id - 1]
                            prev_tokenized_sent = [Token(text=tk.text, idx=len(appended_concat_question_text) + tk.idx)
                                                   for tk in prev_tokenized_sent][:self._sent_limit]
                            prev_tokenized_sent.append(Token(text='[SEP]', idx=prev_tokenized_sent[-1].idx + 1))
                            prev_sent_text = "{}[SEP]".format(cur_para[sent_id - 1])

                        else:
                            prev_sent_text = ""
                            prev_tokenized_sent = []

                        tokenized_sent = [Token(text=tk.text, idx=len(appended_concat_question_text)
                                                                  + len(prev_sent_text) + tk.idx)
                                          for tk in tokenized_sent][:self._sent_limit]
                        tokenized_sent.append(Token(text='[SEP]', idx=tokenized_sent[-1].idx + 1))
                        appended_sent_text = "{}[SEP]".format(sent)
                        tokenized_concat_ques.extend(prev_tokenized_sent)
                        tokenized_concat_ques.extend(tokenized_sent)
                        sent_offset.extend([(tk.idx, tk.idx + len(tk.text)) for tk in prev_tokenized_sent])
                        sent_offset.extend([(tk.idx, tk.idx + len(tk.text)) for tk in tokenized_sent])
                        sent_start = tokenized_sent[0].idx
                        sent_end = tokenized_sent[-1].idx
                    else:
                        next_tokenized_sent = tokenized_sents[para_id][sent_id + 1]
                        next_tokenized_sent = [Token(text=tk.text) for tk in next_tokenized_sent][:self._context_limit]
                        next_tokenized_sent.append(Token(text='[SEP]'))
                        prev_tokenized_sent = tokenized_sents[para_id][sent_id - 1]
                        prev_tokenized_sent = [Token(text=tk.text, idx=len(appended_concat_question_text) + tk.idx)
                                               for tk in prev_tokenized_sent][:self._sent_limit]
                        prev_tokenized_sent.append(Token(text='[SEP]', idx=prev_tokenized_sent[-1].idx + 1))
                        prev_sent_text = "{}[SEP]".format(cur_para[sent_id - 1])
                        tokenized_sent = [Token(text=tk.text, idx=len(appended_concat_question_text)
                                          + len(prev_sent_text) + tk.idx)
                                          for tk in tokenized_sent][:self._sent_limit]
                        tokenized_sent.append(Token(text='[SEP]', idx=tokenized_sent[-1].idx + 1))
                        appended_sent_text = "{}[SEP]".format(sent)

                        tokenized_concat_ques.extend(prev_tokenized_sent)
                        tokenized_concat_ques.extend(tokenized_sent)
                        tokenized_concat_ques.extend(next_tokenized_sent)
                        sent_offset.extend([(tk.idx, tk.idx + len(tk.text)) for tk in prev_tokenized_sent])
                        sent_offset.extend([(tk.idx, tk.idx + len(tk.text)) for tk in tokenized_sent])
                        sent_start = tokenized_sent[0].idx
                        sent_end = tokenized_sent[-1].idx

                    tokenized_sent = tokenized_concat_ques
                    # print('tokenized_sent:', tokenized_sent)
                    sent = '{}{}'.format(appended_concat_question_text, sent)
                else:
                    tokenized_sent.insert(0, Token(text='[CLS]', idx=0))
                    tokenized_sent.append(Token(text='[SEP]', idx=6))
                    sent_offset.extend([(tk.idx, tk.idx + len(tk.text)) for tk in tokenized_sent])
                    sent_start = 0
                    sent_end = 5

                # sent_start = tokenized_sent[0].idx
                # sent_end = tokenized_sent[-1].idx
                passage_sent_tokens.append(tokenized_sent)

                concat_article += sent
                sent_starts.append(sent_start)
                sent_ends.append(sent_end)
                sents_offset.append(sent_offset)

                # print(sent_start, sent_end)
                # print(sent_offset)
                # print(sent)
                # input()
                if (cur_title, sent_id) in sp_set:
                    if answer_text in sent:
                        ans_sent_idxs.append(len(sent_labels))
                    supporting_facts.append(sent)
                    sent_labels.append(1)
                else:
                    sent_labels.append(0)

        return (tokenized_stand_alone_ques,
                appended_question_text,
                concat_article,
                sent_labels,
                [answer_text],
                passage_sent_tokens,
                evd_possible_chains,
                ans_sent_idxs,
                zip(sent_starts, sent_ends),
                sents_offset,
                article_id)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        if self._filter_compare_q:
            # filter out instances whose answer is ``yes`` or ``no``
            dataset = [d for d in dataset if not d['answer'] in ['yes', 'no']]
            # filter out instances whose answer is in question
            dataset = [d for d in dataset if not d['answer'] in d['question']]
        logger.info("Reading the dataset")

        for article in dataset:
            paragraphs = article['context']
            if len(paragraphs) >= 8 and self.training:
                continue
            processed_article = self.process_raw_instance(article)
            instance = self.text_to_instance(*processed_article)
            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokenized_stand_alone_ques: List[Token],
                         question_text: str,
                         passage_text: str,
                         sent_labels: List[int] = None,
                         answer_texts: List[str] = None,
                         passage_sent_tokens: List[List[Token]] = None,
                         evd_possible_chains: List[List[int]] = None,
                         ans_sent_idxs: List[int] = None,
                         sents_span: List[Tuple[int, int]] = None,
                         sents_offset: List[Tuple] = None,
                         article_id: str = None) -> Instance:

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans_sent: List[Tuple[int, int]] = []
        for i, char_span_sent in enumerate(sents_span):
            char_sent_start, char_sent_end = char_span_sent[0], char_span_sent[1]
            sent_offset = sents_offset[i]
            total_wordpiece = 0

            try:
                (span_start, span_end), error = util.char_span_to_token_span(sent_offset,
                                                                             (char_sent_start, char_sent_end))
                for j, token in enumerate(passage_sent_tokens[i]):
                    total_wordpiece += len(wordpiece_tokenizer(token.text.lower()))
                    if total_wordpiece >= self._word_piece_limit:
                        break
                if span_end >= j or span_start >= j:
                    span_start = j-1 if span_start >= j else span_start
                    span_end = j-1 if span_end >= j else span_end
                    # print(passage_sent_tokens[i])
                    # print("span start and end:", span_start, span_end+1)
                    # print("tokens:", passage_sent_tokens[i][span_start: span_end])
                    # print('overflow:', j)
                    # print('total wordpiece:', total_wordpiece)
                    # input()
            except IndexError:
                print(sent_offset)
                print(char_sent_start, char_sent_end)
                input()
            token_spans_sent.append((span_start, span_end))
            # print(span_start, span_end+1)
            # print(passage_sent_tokens[i][span_start: span_end])
            # input()

        return make_reading_comprehension_instance(tokenized_stand_alone_ques,
                                                   passage_sent_tokens,
                                                   self._token_indexers,
                                                   passage_text,
                                                   sent_labels,
                                                   answer_texts,
                                                   evd_possible_chains,
                                                   ans_sent_idxs,
                                                   token_spans_sent,
                                                   article_id,
                                                   para_limit=self._para_limit)


if __name__ == '__main__':
    reader = HotpotDatasetReader()
    reader.read("/backup2/jfchen/data/hotpot/hotpot_test.json")
