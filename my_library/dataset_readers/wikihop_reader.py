import json
import logging
import numpy as np
import re
from itertools import combinations, product
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


def process_sent_spans(token_spans_sent, sent_labels, passage_field, para_limit):
    sent_spans: List[Field] = []
    sent_labels_: List[Field] = []
    if token_spans_sent:
        for (start, end), label in zip(token_spans_sent, sent_labels):
            if start < para_limit and end < para_limit:
                sent_spans.append(SpanField(start, end, passage_field))
                sent_labels_.append(LabelField(label, skip_indexing=True))
            elif start < para_limit and end >= para_limit:
                sent_spans.append(SpanField(start, para_limit - 1, passage_field))
                sent_labels_.append(LabelField(label, skip_indexing=True))
    return sent_spans, sent_labels_


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
                chain = [s_idx+1 for s_idx in chain] + [0]
            evd_possible_chains_.append(chain)
        fields['evd_chain_labels'] = ListField([ArrayField(np.array(ch), padding_value=0) 
                                                for ch in evd_possible_chains_])
    return evd_possible_chains_


def process_answer_spans(token_spans, token_spans_sp, answer_texts, passage_field, para_limit, fields):
    # if yes/no question, set span_start, span_end to IGNORE_INDEX
    if answer_texts is None:
        pass
    elif answer_texts[0] == 'yes':
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


def make_meta_data(passage_text, passage_offsets, question_tokens, passage_tokens, token_spans_sp, token_spans_sent,
                   sent_labels, answer_texts, evd_possible_chains, evd_possible_chains_, ans_sent_idxs, article_id):
    # 0 denotes eos, shifts by one
    if not ans_sent_idxs is None:
        ans_sent_idxs = [s_idx+1 for s_idx in ans_sent_idxs if s_idx < len(sent_labels)]
    metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                'question_tokens': [token.text for token in question_tokens],
                'passage_tokens': [token.text for token in passage_tokens],
                'token_spans_sp': token_spans_sp,
                'token_spans_sent': token_spans_sent,
                'sent_labels': sent_labels,
                '_id': article_id}
    if answer_texts:
        metadata['answer_texts'] = answer_texts
    if not evd_possible_chains is None:
        metadata['evd_possible_chains'] = evd_possible_chains_
    if not ans_sent_idxs is None:
        metadata['ans_sent_idxs'] = ans_sent_idxs
    return metadata


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_tokens: List[Token],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        token_spans: List[Tuple[int, int]] = None,
                                        token_spans_sp: List[Tuple[int, int]] = None,
                                        token_spans_sent: List[Tuple[int, int]] = None,
                                        sent_labels: List[int] = None,
                                        answer_texts: List[str] = None,
                                        passage_offsets: List[Tuple] = None,
                                        evd_possible_chains: List[List[int]] = None,
                                        ans_sent_idxs: List[int] = None,
                                        article_id: str = None,
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
    limit = len(passage_tokens) if para_limit > len(passage_tokens) else para_limit
    passage_tokens = passage_tokens[:limit]
    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    # sent_spans: list of [SpanField[sent_start, sent_end]], denote the start and end offset for each sentence
    # sent_labels_: list of [LabelField[label]], denote the whether a sentence is a supporting fact
    sent_spans, sent_labels_ = process_sent_spans(token_spans_sent, sent_labels, passage_field, para_limit)
    fields['sent_labels'] = ListField(sent_labels_)
    fields['sentence_spans'] = ListField(sent_spans)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)

    # filter spans that exceed para limit so that the info in metadata is correct
    token_spans_sent = [(s, e if e < limit else limit - 1) for s, e in token_spans_sent if s < limit]
    token_spans_sp = [(s, e if e < limit else limit - 1) for s, e in token_spans_sp if s < limit]
    sent_labels = sent_labels[:len(token_spans_sent)]
    process_answer_spans(token_spans, token_spans_sp, answer_texts, passage_field, para_limit, fields)
    evd_possible_chains_ = process_evidence_chains(evd_possible_chains, sent_labels_, fields)

    metadata = make_meta_data(passage_text, passage_offsets, question_tokens, passage_tokens, token_spans_sp,
                              token_spans_sent, sent_labels, answer_texts, evd_possible_chains, evd_possible_chains_,
                              ans_sent_idxs, article_id)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)


@DatasetReader.register("wikihop_reader")
class WikihopDatasetReader(DatasetReader):
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
                 sent_limit: int = 80,
                 filter_compare_q: bool = False,
                 chain: str = 'rb',
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._para_limit = para_limit
        self._sent_limit = sent_limit
        self._filter_compare_q = filter_compare_q
        self.chain = chain


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
        paragraphs = article['passage']
        if self.chain == 'rb':
            evd_possible_chains = article.get("possible_chain", None)
            evd_possible_chains = [evd_possible_chains] if evd_possible_chains is not None else None
        elif self.chain == 'nn':
            evd_possible_chains = article.get("pred_chains", None)

        answer_text = article['answer'].strip().replace("\n", "") if 'answer' in article else None
        concat_article = ""
        passage_tokens = []
        # in wikihop, we treat the sentences in the chain produces by the rb algorithm as supporting facts
        supporting_facts = []
        # passage_offsets, used to convert char-span to token-span
        passage_offsets = []
        sent_starts = []
        sent_ends = []
        # labels denoting whether a sentence is a supporting fact or not
        sent_labels = []
        # the list storing the indices of the sentences that contain the answer
        ans_sent_idxs = []

        for para in paragraphs:
            for sent_id, sent in enumerate(para):
                tokenized_sent = self._tokenizer.tokenize(sent)
                tokenized_sent = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_sent]
                sent_offset = [(tk.idx + len(concat_article),
                                tk.idx + len(tk.text) + len(concat_article)) for tk in tokenized_sent]
                if sent_offset:
                    if answer_text and answer_text in sent.lower():
                        ans_sent_idxs.append(len(sent_labels))
                    if evd_possible_chains and len(sent_labels) in evd_possible_chains[0]:
                        supporting_facts.append(sent)
                        sent_labels.append(1)
                    else:
                        sent_labels.append(0)
                    sent_start = sent_offset[0][0]
                    sent_end = sent_offset[-1][1]
                    sent_starts.append(sent_start)
                    sent_ends.append(sent_end)
                passage_offsets.extend(sent_offset)
                concat_article += sent
                passage_tokens.extend(tokenized_sent)

        question_text = article['question'].strip().replace("\n", "")
        if answer_text:
            span_starts = self.find_all_span_starts(answer_text, concat_article)
            span_ends = [start + len(answer_text) for start in span_starts]
        else:
            span_starts = None
            span_ends = None
        sp_starts = [self.find_span_starts(s, concat_article) for s in supporting_facts]
        sp_ends = [start + len(span) for span, start in zip(supporting_facts, sp_starts)]

        return (question_text,
                concat_article,
                zip(span_starts, span_ends) if span_starts else None,
                zip(sp_starts, sp_ends),
                zip(sent_starts, sent_ends),
                sent_labels,
                [answer_text] if answer_text else None,
                passage_tokens,
                passage_offsets,
                evd_possible_chains,
                ans_sent_idxs,
                article_id)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)[:100]
        if self._filter_compare_q:
            # filter out instances whose answer is ``yes`` or ``no``
            dataset = [d for d in dataset if not d['answer'] in ['yes', 'no']]
            # filter out instances whose answer is in question
            dataset = [d for d in dataset if not d['answer'] in d['question']]
        logger.info("Reading the dataset")

        for article in dataset:
            processed_article = self.process_raw_instance(article)
            instance = self.text_to_instance(*processed_article)
            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         char_spans_sp: List[Tuple[int, int]] = None,
                         char_spans_sent: List[Tuple[int, int]] = None,
                         sent_labels: List[int] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None,
                         passage_offsets: List[Tuple] = None,
                         evd_possible_chains: List[List[int]] = None,
                         ans_sent_idxs: List[int] = None,
                         article_id: str = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)

        char_spans = char_spans or []
        char_spans_sp = char_spans_sp or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        token_spans_sp: List[Tuple[int, int]] = []
        token_spans_sent: List[Tuple[int, int]] = []

        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        for char_span_sp_start, char_span_sp_end in char_spans_sp:
            (span_start_sp, span_end_sp), error = util.char_span_to_token_span(passage_offsets,
                                                                               (char_span_sp_start, char_span_sp_end))
            token_spans_sp.append((span_start_sp, span_end_sp))

        for char_span_sent_start, char_span_sent_end in char_spans_sent:
            (span_start_sent, span_end_sent), error = util.char_span_to_token_span(passage_offsets,
                                                                                   (char_span_sent_start,
                                                                                    char_span_sent_end))
            token_spans_sent.append((span_start_sent, span_end_sent))

        tokenized_ques = self._tokenizer.tokenize(question_text)
        tokenized_ques = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_ques]

        return make_reading_comprehension_instance(tokenized_ques,
                                                   passage_tokens,
                                                   self._token_indexers,
                                                   passage_text,
                                                   token_spans,
                                                   token_spans_sp,
                                                   token_spans_sent,
                                                   sent_labels,
                                                   answer_texts,
                                                   passage_offsets,
                                                   evd_possible_chains,
                                                   ans_sent_idxs,
                                                   article_id,
                                                   para_limit=self._para_limit)


if __name__ == '__main__':
    reader = WikihopDatasetReader()
    reader.read("/scratch/cluster/jfchen/jfchen/data/wikihop/train_chain/train0.json")
