import json
import logging
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from collections import Counter
from overrides import overrides
from allennlp.data.fields import Field, TextField, IndexField, ArrayField, SpanField, \
    MetadataField, LabelField, ListField, SequenceLabelField
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from spacy.tokens import Token
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_tokens: List[Token],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        token_spans: List[Tuple[int, int]] = None,
                                        token_spans_sp: List[Tuple[int, int]] = None,
                                        token_spans_sent: List[Tuple[int, int]] = None,
                                        answer_texts: List[str] = None,
                                        passage_offsets: List[Tuple] = None,
                                        passage_dep_heads: List[int] = None,
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

    limit = len(passage_tokens) if para_limit > len(passage_tokens) else para_limit
    passage_tokens = passage_tokens[:limit]
    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    sent_spans: List[Field] = []
    if token_spans_sent:
        for start, end in token_spans_sent:
            if start < para_limit and end < para_limit:
                sent_spans.append(SpanField(start, end, passage_field))
    fields['sentence_spans'] = ListField(sent_spans)

    metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                'question_tokens': [token.text for token in question_tokens],
                'passage_tokens': [token.text for token in passage_tokens]}
    if answer_texts:
        metadata['answer_texts'] = answer_texts

    # print('answer:', answer_texts[0])
    # print('answer_text:', answer_texts)
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
            # There may be multiple answer annotations, so we pick the one that occurs the most.  This
            # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
            # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
            # all of the annotations.  This is why we have a separate official SQuAD metric calculation
            # (the "em" and "f1" metrics use the official script).
            candidate_answers: Counter = Counter()
            for span_start, span_end in token_spans:
                candidate_answers[(span_start, span_end)] += 1
            span_start, span_end = candidate_answers.most_common(1)[0][0]
            # print('best span:', span_start, span_end)
            # print('span:', span_start, span_end)
            # print(metadata['passage_tokens'][span_start:span_end + 1])
            if span_start > para_limit or span_end > para_limit:
                # print('span_start, span_end:', span_start, span_end)
                fields['span_start'] = IndexField(-100, passage_field)
                fields['span_end'] = IndexField(-100, passage_field)
            else:
                fields['span_start'] = IndexField(span_start, passage_field)
                fields['span_end'] = IndexField(span_end, passage_field)
        else:
            fields['span_start'] = IndexField(-100, passage_field)
            fields['span_end'] = IndexField(-100, passage_field)

    # print('fields:', fields['span_start'], fields['span_end'], fields['q_type'])

    if token_spans_sp:
        sp_mask = np.zeros(len(passage_tokens))
        for s, e in token_spans_sp:
            sp_mask[s:e] = 1
    else:
        sp_mask = np.ones(len(passage_tokens))

    if passage_dep_heads:
        dep_mask = np.zeros((len(passage_tokens), len(passage_tokens)))
        valid_heads = [h for h in passage_dep_heads[:limit] if 0 <= h < limit]
        valid_childs = [i for i, h in enumerate(passage_dep_heads[:limit]) if 0 <= h < limit]
        dep_mask[valid_heads+valid_childs, valid_childs+valid_heads] = 1
    else:
        dep_mask = np.ones((len(passage_tokens), len(passage_tokens)))

    fields['sp_mask'] = ArrayField(sp_mask)
    fields['dep_mask'] = ArrayField(dep_mask)
    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)


@DatasetReader.register("hotpot_reader")
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
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._para_limit = para_limit
        self.count = 0

    @staticmethod
    def find_all_span_starts(answer, context):
        return [m.start() for m in re.finditer(re.escape(answer), context)]

    @staticmethod
    def find_span_starts(span, context):
        return re.search(re.escape(span), context).start()

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")

        for article in dataset:
            paragraphs = article['context']
            dependency_paragraphs = article['golden_head']
            concat_article = ""
            passage_tokens = []
            supporting_facts = []
            passage_offsets = []
            sent_starts = []
            sent_ends = []
            sp_set = set(list(map(tuple, article['supporting_facts'])))
            passage_dep_heads = []

            for para, dep_para in zip(paragraphs, dependency_paragraphs):
                cur_title, cur_para = para[0], para[1]
                dep_title, cur_dep_para = dep_para[0], dep_para[1]
                assert cur_title == dep_title, "Not equal: %s, %s" % (cur_title, dep_title)
                for sent_id, (sent, dep_heads) in enumerate(zip(cur_para, cur_dep_para)):
                    # heads are 1-indexing, so shifted by 1 and add the sentence offset
                    dep_heads = [h-1+len(passage_tokens) if h > 0 else -1 for h in dep_heads]
                    passage_dep_heads.extend(dep_heads)

                    tokenized_sent = self._tokenizer.tokenize(sent)
                    sent_offset = [(tk.idx + len(concat_article),
                                    tk.idx + len(tk.text) + len(concat_article)) for tk in tokenized_sent]
                    if sent_offset:
                        sent_start = sent_offset[0][0]
                        sent_end = sent_offset[-1][1]
                        # sent_start = tokenized_sent[0].idx + len(concat_article)
                        # sent_end = sent_start + len(sent) - 1
                        sent_starts.append(sent_start)
                        sent_ends.append(sent_end)
                    passage_offsets.extend(sent_offset)
                    concat_article += sent
                    passage_tokens.extend(tokenized_sent)
                    if (cur_title, sent_id) in sp_set:
                        supporting_facts.append(sent)

            question_text = article['question'].strip().replace("\n", "")
            answer_text = article['answer'].strip().replace("\n", "")
            span_starts = self.find_all_span_starts(answer_text, concat_article)
            # print('article id:', article['_id'])
            # print('span_starts:', span_starts)
            if not span_starts:
                # print(self.count)
                self.count += 1
            span_ends = [start + len(answer_text) for start in span_starts]
            # print('span_ends:', span_ends)
            sp_starts = [self.find_span_starts(s, concat_article) for s in supporting_facts]
            sp_ends = [start + len(span) for span, start in zip(supporting_facts, sp_starts)]

            instance = self.text_to_instance(question_text,
                                             concat_article,
                                             zip(span_starts, span_ends),
                                             zip(sp_starts, sp_ends),
                                             zip(sent_starts, sent_ends),
                                             [answer_text],
                                             passage_tokens,
                                             passage_offsets,
                                             passage_dep_heads)
            # print('supporting_facts:', supporting_facts)
            # print(instance)
            # print(instance["span_start"])
            # print(instance["span_end"])
            # print(instance['metadata'].metadata)
            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         char_spans_sp: List[Tuple[int, int]] = None,
                         char_spans_sent: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None,
                         passage_offsets: List[Tuple] = None,
                         passage_dep_heads: List[int] = None) -> Instance:
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
            # print("char answer:", passage_text[char_span_start:char_span_end])
            # print("Answer:", passage_tokens[span_start:span_end+1])
            # print(answer_texts)
        for char_span_sp_start, char_span_sp_end in char_spans_sp:
            (span_start_sp, span_end_sp), error = util.char_span_to_token_span(passage_offsets,
                                                                               (char_span_sp_start, char_span_sp_end))
            token_spans_sp.append((span_start_sp, span_end_sp))

        for char_span_sent_start, char_span_sent_end in char_spans_sent:
            (span_start_sent, span_end_sent), error = util.char_span_to_token_span(passage_offsets,
                                                                            (char_span_sent_start, char_span_sent_end))
            token_spans_sent.append((span_start_sent, span_end_sent))

        return make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                   passage_tokens,
                                                   self._token_indexers,
                                                   passage_text,
                                                   token_spans,
                                                   token_spans_sp,
                                                   token_spans_sent,
                                                   answer_texts,
                                                   passage_offsets,
                                                   passage_dep_heads,
                                                   para_limit=self._para_limit)


if __name__ == '__main__':
    reader = HotpotDatasetReader()
    reader.read("/backup2/jfchen/data/hotpot/hotpot_test.json")
