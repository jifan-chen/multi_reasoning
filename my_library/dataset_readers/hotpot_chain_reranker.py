import json
import logging
import re
from rouge import Rouge
from typing import Dict, List, Tuple, Any
import numpy as np
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


def make_reading_comprehension_instance(question_text: str,
                                        passage_text: str,
                                        answer_text: str,
                                        label: float,
                                        question_passage_tokens: List[Token],
                                        question_passage_offsets: List[Tuple[int, int]],
                                        token_indexers: Dict[str, TokenIndexer],
                                        id: str = None,
                                        pred_chains: List[Tuple[List, float]] = None,
                                        sp_facts_id: List[int] = None,
                                        article: Dict = None) -> Instance:
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
    fields: Dict[str, Field] = {}
    # This is separate so we can reference it later with a known type.
    question_passage_field = TextField(question_passage_tokens, token_indexers)

    fields['question_passage'] = question_passage_field
    fields['label'] = LabelField(label, skip_indexing=True)

    metadata = {'original_passage': passage_text, 'token_offsets': question_passage_offsets,
                'question_text': question_text, 'original_label_score': label, 'id': id, 'pred_chains': pred_chains,
                'sp_set_id': sp_facts_id, 'original_article': article,
                'passage_tokens': [token.text for token in question_passage_tokens]}

    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)


@DatasetReader.register("hotpot_bert_reranker")
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
                 sent_limit: int = 80,
                 rerank_by_sp: bool = False,
                 validation: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._para_limit = para_limit
        self._sent_limit = sent_limit
        self._rerank_by_sp = rerank_by_sp
        self._validation = validation
        self.count = 0

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
            combined_chains_text = ''
            for e in chain:
                try:
                    chain_text = all_sents[e]
                except IndexError:
                    continue
                combined_chains_text += chain_text

                if answer_text in chain_text:
                    answer_count += 1
            f1_avg += self.f1_score_by_question(combined_chains_text, question_text)
            f1_avg += answer_count
            reranked_chains.append([chain, f1_avg])
        reranked_chains = sorted(reranked_chains, key=lambda rc: rc[1], reverse=True)
        sample_idxs = [0, 1, -1, -2]
        sampled_chains = []
        for i in sample_idxs:
            chain = reranked_chains[i]
            label = 1 if chain[1] >= 1 else 0
            sampled_chains.append([chain[0], label])
        return sampled_chains

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
        pred_chains = []
        concat_qp = ""
        answer_in = []
        question_text = article['question'].strip().replace("\n", "")
        answer_text = article['answer'].strip().replace("\n", "")
        sp_set = set(list(map(tuple, article['supporting_facts'])))
        all_sents, sp_facts_id, sent_labels = self.preprocess_global_info(paragraphs, sp_set)

        # pred_chains = self.chain_rerank(article['pred_chains'], sp_facts_id) if self._rerank_by_sp \
        #     else article['pred_chains']
        top_k = 10
        self._rerank_by_sp = False
        if self._rerank_by_sp:
            pred_chains = self.chain_rerank_by_question(article['pred_chains'], question_text, answer_text, all_sents)
        else:
            for chain in article['pred_chains']:
                for id in chain:
                    try:
                        if answer_text in all_sents[id]:
                            pred_chains.append([chain, 1])
                        else:
                            pred_chains.append([chain, 0])
                    except IndexError:
                        pred_chains.append([chain, 0])

        # print(self._rerank_by_sp)
        # print(pred_chains)
        pred_chains = pred_chains[:top_k]
        # print(pred_chains)
        tokenized_ques = self._tokenizer.tokenize(question_text)
        tokenized_ques = [Token(text=tk.text, idx=tk.idx + 1) for tk in tokenized_ques]
        tokenized_ques.insert(0, Token(text='[CLS]', idx=0))
        tokenized_ques.append(Token(text='[SEP]', idx=tokenized_ques[-1].idx + 1))
        appended_question_text = "[CLS]{}[SEP]".format(question_text)
        sent_offset = [(tk.idx + len(concat_qp),
                        tk.idx + len(tk.text) + len(concat_qp)) for tk in tokenized_ques]
        # question_passage_offsets.extend(sent_offset)
        # concat_qp += appended_question_text
        # question_passage_tokens.extend(tokenized_ques)
        instances = []

        if not self._validation and all([c[1] == 0 for c in pred_chains]):
            return instances
        # print(pred_chains)
        for chain in pred_chains:
            question_passage_tokens = []
            question_passage_offsets = []
            concat_qp = ""
            concat_qp += appended_question_text
            question_passage_tokens.extend(tokenized_ques)
            question_passage_offsets.extend(sent_offset)
            label = chain[1]
            # print(chain)
            if len(chain[0]) > 0:
                for sent_id in chain[0]:
                    try:
                        sent = all_sents[sent_id]
                    except IndexError:
                        continue
                    tokenized_sent = self._tokenizer.tokenize(sent)
                    combined_chains.append(sent)
                    tokenized_sent = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_sent]

                    concat_qp += sent
                    question_passage_tokens.extend(tokenized_sent)

            concat_qp += '[SEP]'
            question_passage_tokens.append(Token(text='[SEP]', idx=question_passage_tokens[-1].idx + 1))
            question_passage_offsets.append((len(concat_qp), len(concat_qp) + len('[SEP]')))
            # print(chain)
            # print(label)
            # print(answer_text)
            # input()
            instances.append(self.text_to_instance(question_text,
                                                   concat_qp,
                                                   answer_text,
                                                   label,
                                                   question_passage_tokens,
                                                   question_passage_offsets,
                                                   article_id,
                                                   chain,
                                                   sp_facts_id,
                                                   article))
        return instances

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
            for p in processed_article:
                yield p

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         answer_texts: str = None,
                         f1_score: float = None,
                         question_passage_tokens: List[Token] = None,
                         question_passage_offsets: List[Tuple[int, int]] = None,
                         article_id: str = None,
                         pred_chains: List[Tuple[List, float]] = None,
                         sp_facts_id: List[float] = None,
                         article: Dict = None) -> Instance:

        return make_reading_comprehension_instance(question_text,
                                                   passage_text,
                                                   answer_texts,
                                                   f1_score,
                                                   question_passage_tokens,
                                                   question_passage_offsets,
                                                   self._token_indexers,
                                                   article_id,
                                                   pred_chains,
                                                   sp_facts_id,
                                                   article)


if __name__ == '__main__':
    reader = HotpotDatasetReader()
    reader.read("/backup2/jfchen/data/hotpot/hotpot_test.json")
