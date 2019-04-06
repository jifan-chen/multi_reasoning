import json
import logging
from typing import Dict, List, Tuple, Optional

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
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_tokens: List[Token],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        token_spans_sent: List[Tuple[int, int]] = None,
                                        sent_labels: List[int] = None,
                                        answer_texts: List[str] = None,
                                        passage_offsets: List[Tuple] = None,) -> Instance:
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
    fields: Dict[str, Field] = {}

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    sent_spans: List[Field] = []
    sent_labels_: List[Field] = []
    if token_spans_sent:
        for (start, end), label in zip(token_spans_sent, sent_labels):
            sent_spans.append(SpanField(start, end, passage_field))
            sent_labels_.append(LabelField(label, skip_indexing=True))

    fields['sent_labels'] = ListField(sent_labels_)
    fields['sentence_spans'] = ListField(sent_spans)

    # filter spans that exceed para limit so that the info in metadata is correct
    sent_labels = sent_labels[:len(token_spans_sent)]

    metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                'question_tokens': [token.text for token in question_tokens],
                'passage_tokens': [token.text for token in passage_tokens],
                'token_spans_sent': token_spans_sent,
                'sent_labels': sent_labels}
    if answer_texts:
        metadata['answer_texts'] = answer_texts

    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)


@DatasetReader.register("wikihop_reader")
class WikihopReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.
    We also support limiting the maximum length for both passage and question. However, some gold
    answer spans may exceed the maximum passage length, which will cause error in making instances.
    We simply skip these spans to avoid errors. If all of the gold answer spans of an example
    are skipped, during training, we will skip this example. During validating or testing, since
    we cannot skip examples, we use the last token as the pseudo gold answer span instead. The
    computed loss will not be accurate as a result. But this will not affect the answer evaluation,
    because we keep all the original gold answer texts.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    skip_invalid_examples: ``bool``, optional (default=False)
        if this is true, we will skip those invalid examples
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 skip_invalid_examples: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_splitter = SpacySentenceSplitter(rule_based=True)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)

        logger.info("Reading the dataset")
        for article in dataset_json:
            answer_text = article['answer']
            question_text = article['query']
            paragraphs = article['supports']
            concat_article = ""
            sent_labels = []
            sent_starts = []
            sent_ends = []
            passage_offsets = []
            passage_tokens = []
            for para in paragraphs:
                sentences = self._sentence_splitter.split_sentences(para)
                for sent in sentences:
                    tokenized_sent = self._tokenizer.tokenize(sent)
                    tokenized_sent = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_sent]
                    sent_offset = [(tk.idx + len(concat_article),
                                    tk.idx + len(tk.text) + len(concat_article)) for tk in tokenized_sent]
                    passage_offsets.extend(sent_offset)
                    concat_article += sent
                    passage_tokens.extend(tokenized_sent)
                    if sent_offset:
                        sent_start = sent_offset[0][0]
                        sent_end = sent_offset[-1][1]
                        sent_starts.append(sent_start)
                        sent_ends.append(sent_end)

                        if answer_text in sent.lower():
                            sent_labels.append(1)
                        else:
                            sent_labels.append(0)
            # print(sent_labels)
            # print(answer_text)
            # print(question_text)
            # print(concat_article)
            instance = self.text_to_instance(question_text,
                                             concat_article,
                                             zip(sent_starts, sent_ends),
                                             sent_labels,
                                             answer_text,
                                             passage_tokens,
                                             passage_offsets)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans_sent: List[Tuple[int, int]] = None,
                         sent_labels: List[int] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None,
                         passage_offsets: List[Tuple] = None) -> Instance:

        token_spans_sent: List[Tuple[int, int]] = []

        for char_span_sent_start, char_span_sent_end in char_spans_sent:
            (span_start_sent, span_end_sent), error = util.char_span_to_token_span(passage_offsets,
                                                                                   (char_span_sent_start,
                                                                                    char_span_sent_end))
            token_spans_sent.append((span_start_sent, span_end_sent))

        tokenized_ques = self._tokenizer.tokenize(" ".join(question_text.split('_')))
        tokenized_ques = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_ques]
        # print(tokenized_ques)
        if len(passage_tokens) > 2250:
            return None
        else:
            return make_reading_comprehension_instance(tokenized_ques,
                                                       passage_tokens,
                                                       self._token_indexers,
                                                       passage_text,
                                                       token_spans_sent,
                                                       sent_labels,
                                                       answer_texts,
                                                       passage_offsets)
