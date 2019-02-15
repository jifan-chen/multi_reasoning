from torch.autograd import Variable
import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder, MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


@Model.register("hotpot")
class BidirectionalAttentionFlow(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).
    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 type_encoder: Seq2SeqEncoder,
                 self_attention_layer: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 strong_sup: bool = False,
                 # initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = matrix_attention
        self._modeling_layer = modeling_layer
        self._span_end_encoder = span_end_encoder
        self._self_attention_layer = self_attention_layer
        self._span_start_encoder = span_start_encoder
        self._type_encoder = type_encoder
        self._strong_sup = strong_sup
        self.biattention = BiAttention(input_size=200, dropout=0.2)

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        projection_dim = modeling_dim
        self_attention_dim = self_attention_layer.get_output_dim()

        self._merged_projection = torch.nn.Sequential(torch.nn.Linear(encoding_dim * 4, projection_dim),
                                                      torch.nn.ReLU())

        span_start_input_dim = span_start_encoder.get_output_dim()

        self._span_start_predictor = torch.nn.Linear(span_start_input_dim, 1)
        self._type_linear = torch.nn.Sequential(
                torch.nn.Linear(2 * modeling_dim, modeling_dim),
                torch.nn.ReLU()
            )
        self._type_predictor = torch.nn.Linear(2 * modeling_dim, 3)

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = span_end_encoding_dim
        self._span_end_projection = torch.nn.Linear(projection_dim + self_attention_dim*3, projection_dim)
        self._span_end_predictor = torch.nn.Linear(span_end_input_dim, 1)

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), projection_dim,
                               "modeling layer input dim", "merged_projection_dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")
        check_dimensions_match(span_end_encoder.get_input_dim(), projection_dim + self_attention_dim,
                               "span end encoder input dim", "merged_projection_dim + self_attention_dim")

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        # initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                sp_mask: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        q_type: ''torch.IntTensor'', optional
        sp_mask: ''torch.IntTensor'', optional, mask indicates where the supporting facts appear
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.
        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        # embedded_question = self._highway_layer(self._text_field_embedder(question))
        # embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        # passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # # Shape: (batch_size, passage_length, question_length)
        # passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # # Shape: (batch_size, passage_length, encoding_dim)
        # passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)
        #
        # # We replace masked values with something really negative here, so they don't affect the
        # # max below.
        # masked_similarity = util.replace_masked_values(passage_question_similarity,
        #                                                question_mask.unsqueeze(1),
        #                                                -1e7)
        # # Shape: (batch_size, passage_length)
        # question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # # Shape: (batch_size, passage_length)
        # question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # # Shape: (batch_size, encoding_dim)
        # question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # # Shape: (batch_size, passage_length, encoding_dim)
        # tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
        #                                                                            passage_length,
        #                                                                             encoding_dim)
        # # Shape: (batch_size, passage_length, encoding_dim * 4)
        # final_merged_passage = torch.cat([encoded_passage,
        #                                   passage_question_vectors,
        #                                   encoded_passage * passage_question_vectors,
        #                                   encoded_passage * tiled_question_passage_vector],
        #                                  dim=-1)

        final_merged_passage = self.biattention(encoded_passage, encoded_question, question_mask)
        final_merged_passage = self._merged_projection(final_merged_passage)
        # modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        # modeling_dim = modeled_passage.size(-1)

        # Shape: (batch_size, passage_length, self_attention_dim)
        # if self._strong_sup:
        #     self_att_passage = self._self_attention_layer(modeled_passage, passage_lstm_mask, sp_mask)
        #     modeled_passage = modeled_passage + self_att_passage[0]
        #     strong_sup_loss = self_att_passage[1]
        # else:
        #     modeled_passage = modeled_passage + self._self_attention_layer(modeled_passage, passage_lstm_mask)

        encoded_span_start = self._span_start_encoder(final_merged_passage, passage_lstm_mask)

        # Shape: (batch_size, passage_length, merged_projection_dim + self_attention_dim))
        # span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(encoded_span_start).squeeze(-1) - 1e30 * (1 - passage_mask)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # Shape: (batch_size, self_attention_dim)
        # span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, self_attention_dim)
        # tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
        #                                                                            passage_length,
        #                                                                            self_attention_dim)

        # Shape: (batch_size, passage_length, merged_projection_dim + self_attention_dim * 3)
        span_end_representation = torch.cat([encoded_span_start, final_merged_passage], dim=-1)

        # Shape: (batch_size, passage_length, projection_dim)
        # span_end_representation = self._span_end_projection(span_end_representation)

        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._span_end_encoder(span_end_representation, passage_lstm_mask)
        # Shape: (batch_size, passage_length, merged_projection_dim + span_end_encoding_dim)
        # span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        span_end_logits = self._span_end_predictor(encoded_span_end).squeeze(-1) - 1e30 * (1 - passage_mask)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        # span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        # span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        best_span = self.get_best_span(span_start_logits, span_end_logits)

        type_representation = torch.cat([final_merged_passage, encoded_span_end], dim=2)
        type_representation = torch.max(type_representation, 1)[0]
        type_logits = self._type_predictor(type_representation)
        type_predicts = torch.argmax(type_logits, 1)

        output_dict = {
                # "passage_question_attention": passage_question_attention,
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "span_end_logits": span_end_logits,
                "span_end_probs": span_end_probs,
                "best_span": best_span,
                }

        # Compute the loss for training.
        if span_start is not None:
            try:
                loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
                self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
                loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
                self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
                self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
                # print('q_type:', q_type)
                loss += nll_loss(util.masked_log_softmax(type_logits, None), q_type)

                # if self._strong_sup:
                #     loss += strong_sup_loss
                #     print('\n strong_sup_loss', strong_sup_loss)

                output_dict["loss"] = loss

            except RuntimeError:
                print('\n meta_data:', metadata)
                print(span_start_logits.shape)

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            count_yes = 0
            count_no = 0
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                if type_predicts[i] == 1:
                    best_span_string = 'yes'
                    count_yes += 1
                elif type_predicts[i] == 2:
                    best_span_string = 'no'
                    count_no += 1
                else:
                    predicted_span = tuple(best_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    best_span_string = passage_str[start_offset:end_offset]

                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                # print('type:', type_predicts[i])
                # print('answer_text:', answer_texts, 'predict:', best_span_string)
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            # print('yes:', count_yes)
            # print('no:', count_no)

        # print('\n........debugging.........')
        # print(best_span)
        # print('\nspan_start:', span_start.squeeze(-1))
        # print('span_end:', span_end.squeeze(-1))
        # print(output_dict['span_start_probs'])

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score,
                }

    @staticmethod
    def get_answer() -> str:
        pass

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

        span_start_logits = span_start_logits.detach().cpu().numpy()
        span_end_logits = span_end_logits.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span