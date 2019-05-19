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


def make_reading_comprehension_instance(question_passage_tokens: List[Token],
                                        option_tokens: List[List[Token]],
                                        answer_label: int,
                                        token_indexers: Dict[str, TokenIndexer],
                                        answer_texts: List[str] = None,
                                        passage_text: str = None) -> Instance:
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
    # print(question_passage_tokens)
    # input()

    fields['question_passage'] = TextField(question_passage_tokens, token_indexers)
    option_field = [TextField(opt_token, token_indexers) for opt_token in option_tokens]
    fields['option'] = ListField(option_field)
    fields['answer'] = LabelField(answer_label, skip_indexing=True)
    # filter spans that exceed para limit so that the info in metadata is correct

    metadata = {'original_passage': passage_text,
                'question_text': [token.text for token in question_passage_tokens]}
    if answer_texts:
        metadata['answer_texts'] = answer_texts

    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)


@DatasetReader.register("wikihop_bert")
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
                 skip_invalid_examples: bool = False,
                 sent_limit: int = 80) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_splitter = SpacySentenceSplitter(rule_based=True)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples
        self._sent_limit = sent_limit

    @staticmethod
    def preprocess_global_info(pragraphs):
        global_id = 0
        all_sents = []
        for para in pragraphs:
            for sent_id, sent in enumerate(para):
                all_sents.append(sent)
                global_id += 1
        return all_sents

    @staticmethod
    def get_topK_chains(pred_chains, top_k, total_num_chains):
        combined_chains_id = []
        for chain in pred_chains[:top_k]:
            for e in chain:
                if e not in combined_chains_id and len(combined_chains_id) < total_num_chains:
                    combined_chains_id.append(e)
        return combined_chains_id

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
            question_text = article['question']
            pred_chains = article['pred_chains']
            paragraphs = article['passage']
            options = article['candidates']
            all_sents = self.preprocess_global_info(paragraphs)
            combined_chains_id = self.get_topK_chains(pred_chains, 5, 5)
            answer_label = options.index(answer_text)
            option_tokens = [self._tokenizer.tokenize(opt) for opt in options]
            combined_chains = []
            concat_qp = ""
            question_passage_tokens = []
            question_passage_offsets = []

            question_tokens = " ".join(question_text.split("_")).split()

            question_tokens = [Token(text=tk, idx=idx + 1) for idx, tk in enumerate(question_tokens)]
            question_tokens.insert(0, Token(text='[CLS]', idx=0))
            question_tokens.append(Token(text='[SEP]', idx=question_tokens[-1].idx + 1))
            concat_qp += "[CLS]{}[SEP]".format(question_text)
            question_passage_tokens.extend(question_tokens)

            # print(question_tokens)
            # input()
            option_tokens = [[Token(text=tk.text, idx=tk.idx + 1) for tk in opt_tokens] for opt_tokens in option_tokens]
            for opt_tokens in option_tokens:
                opt_tokens.insert(0, Token(text='[CLS]', idx=0))

            sent_lengths = [len(all_sents[sent_id]) < self._sent_limit for sent_id in combined_chains]
            limit = self._sent_limit
            if 1 not in sent_lengths:
                limit = 1000
            # print('answer:', answer_text)
            # print('options:', options)
            for sent_id in combined_chains_id:
                sent = all_sents[sent_id]
                # print(sent)
                tokenized_sent = self._tokenizer.tokenize(sent)
                if len(tokenized_sent) < limit:
                    combined_chains.append(sent)
                    tokenized_sent = [Token(text=tk.text, idx=tk.idx) for tk in tokenized_sent]
                    sent_offset_qp = [(tk.idx + len(concat_qp),
                                       tk.idx + len(tk.text) + len(concat_qp)) for tk in tokenized_sent]

                    question_passage_offsets.extend(sent_offset_qp)
                    concat_qp += sent
                    question_passage_tokens.extend(tokenized_sent)

            concat_qp += '[SEP]'
            question_passage_tokens.append(Token(text='[SEP]', idx=question_passage_tokens[-1].idx + 1))
            question_passage_offsets.append((len(concat_qp), len(concat_qp) + len('[SEP]')))

            instance = self.text_to_instance(question_passage_tokens,
                                             option_tokens,
                                             answer_label,
                                             answer_text,
                                             concat_qp)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_passage_tokens: List[Token],
                         option_tokens: List[List[Token]],
                         answer_label: int,
                         answer_texts: List[str] = None,
                         passage_text: str = None) -> Instance:

        return make_reading_comprehension_instance(question_passage_tokens,
                                                   option_tokens,
                                                   answer_label,
                                                   self._token_indexers,
                                                   answer_texts,
                                                   passage_text)
