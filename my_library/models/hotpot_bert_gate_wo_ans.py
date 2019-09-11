from torch.autograd import Variable
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.nn.functional import nll_loss
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure, SquadEmAndF1, Average
from my_library.models.utils import convert_sequence_to_spans, convert_span_to_sequence
from my_library.metrics import AttF1Measure
from my_library.modules import BiAttention
from my_library.models.hotpot_legacy import SpanGate


@Model.register("hotpot_bert_gate_wo_ans")
class GateBidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 gate_sent_encoder: Seq2SeqEncoder,
                 gate_self_attention_layer: Seq2SeqEncoder,
                 span_gate: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 output_att_scores: bool = True,
                 sent_labels_src: str = 'sp',
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(GateBidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        self._dropout = torch.nn.Dropout(p=dropout)

        self._output_att_scores = output_att_scores

        self._sent_labels_src = sent_labels_src


        self._span_gate = span_gate

        if span_gate._gate_self_att:
            self._gate_sent_encoder = gate_sent_encoder
            self._gate_self_attention_layer = gate_self_attention_layer
        else:
            self._gate_sent_encoder = None
            self._gate_self_attention_layer = None

        self._f1_metrics = F1Measure(1)

        self.evd_ans_metric = Average()

        self._loss_trackers = {'loss': Average()}

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                sentence_spans: torch.IntTensor = None,
                sent_labels: torch.IntTensor = None,
                evd_chain_labels: torch.IntTensor = None,
                q_type: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        if self._sent_labels_src == 'chain':
            batch_size, num_spans = sent_labels.size()
            sent_labels_mask = (sent_labels >= 0).float()
            print("chain:", evd_chain_labels)
            # we use the chain as the label to supervise the gate
            # In this model, we only take the first chain in ``evd_chain_labels`` for supervision,
            # right now the number of chains should only be one too.
            evd_chain_labels = evd_chain_labels[:, 0].long()
            # build the gate labels. The dim is set to 1 + num_spans to account for the end embedding
            # shape: (batch_size, 1+num_spans)
            sent_labels = sent_labels.new_zeros((batch_size, 1+num_spans))
            sent_labels.scatter_(1, evd_chain_labels, 1.)
            # remove the column for end embedding
            # shape: (batch_size, num_spans)
            sent_labels = sent_labels[:, 1:].float()
            # make the padding be -1
            sent_labels = sent_labels * sent_labels_mask + -1. * (1 - sent_labels_mask)

        print('\nBert wordpiece size:', passage['bert'].shape)
        # bert embedding for answer prediction
        # shape: [batch_size, max_q_len, emb_size]
        embedded_question = self._text_field_embedder(question, num_wrapping_dims=0)
        # shape: [batch_size, num_sent, max_sent_len+q_len, embedding_dim]
        embedded_passage = self._text_field_embedder(passage, num_wrapping_dims=1)
        # print('\npassage size:', embedded_passage.shape)
        #embedded_question = self._bert_projection(embedded_question)
        #embedded_passage = self._bert_projection(embedded_passage)
        #print('size embedded_passage:', embedded_passage.shape)
        # mask
        ques_mask = util.get_text_field_mask(question, num_wrapping_dims=0).float()
        context_mask = util.get_text_field_mask(passage, num_wrapping_dims=1).float()

        # gate prediction
        # Shape(gate_logit): (batch_size * num_spans, 2)
        # Shape(gate): (batch_size * num_spans, 1)
        # Shape(pred_sent_probs): (batch_size * num_spans, 2)
        # Shape(gate_mask): (batch_size, num_spans)
        #gate_logit, gate, pred_sent_probs = self._span_gate(spans_rep_sp, spans_mask)
        gate_logit, gate, pred_sent_probs, gate_mask, g_att_score = self._span_gate(embedded_passage, context_mask,
                                                                         self._gate_self_attention_layer,
                                                                         self._gate_sent_encoder)
        batch_size, num_spans, max_batch_span_width = context_mask.size()

        loss = F.nll_loss(F.log_softmax(gate_logit, dim=-1).view(batch_size * num_spans, -1),
                          sent_labels.long().view(batch_size * num_spans), ignore_index=-1)

        gate = (gate >= 0.3).long()
        gate = gate.view(batch_size, num_spans)

        output_dict = {
            "pred_sent_labels": gate, #[B, num_span]
            "gate_probs": pred_sent_probs[:, 1].view(batch_size, num_spans), #[B, num_span]
        }
        if self._output_att_scores:
            if not g_att_score is None:
                output_dict['evd_self_attention_score'] = g_att_score

        # Compute the loss for training.
        try:
            #loss = strong_sup_loss
            self._loss_trackers['loss'](loss)
            output_dict["loss"] = loss
        except RuntimeError:
            print('\n meta_data:', metadata)
            print(span_start_logits.shape)

        print("sent label:")
        for b_label in np.array(sent_labels.cpu()):
            b_label = b_label == 1
            indices = np.arange(len(b_label))
            print(indices[b_label] + 1)
        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['answer_texts'] = []
            question_tokens = []
            passage_tokens = []
            #token_spans_sp = []
            #token_spans_sent = []
            sent_labels_list = []
            evd_possible_chains = []
            ans_sent_idxs = []
            ids = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_sent_tokens'])
                #token_spans_sp.append(metadata[i]['token_spans_sp'])
                #token_spans_sent.append(metadata[i]['token_spans_sent'])
                sent_labels_list.append(metadata[i]['sent_labels'])
                ids.append(metadata[i]['_id'])
                passage_str = metadata[i]['original_passage']
                #offsets = metadata[i]['token_offsets']
                answer_texts = metadata[i].get('answer_texts', [])
                output_dict['answer_texts'].append(answer_texts)

                # shift sentence indice back
                evd_possible_chains.append([s_idx-1 for s_idx in metadata[i]['evd_possible_chains'][0] if s_idx > 0])
                ans_sent_idxs.append([s_idx-1 for s_idx in metadata[i]['ans_sent_idxs']])
                if len(metadata[i]['ans_sent_idxs']) > 0:
                    pred_sent_gate = gate[i].detach().cpu().numpy()
                    if any([pred_sent_gate[s_idx-1] > 0 for s_idx in metadata[i]['ans_sent_idxs']]):
                        self.evd_ans_metric(1)
                    else:
                        self.evd_ans_metric(0)
            self._f1_metrics(pred_sent_probs, sent_labels.view(-1), gate_mask.view(-1))
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_sent_tokens'] = passage_tokens
            #output_dict['token_spans_sp'] = token_spans_sp
            #output_dict['token_spans_sent'] = token_spans_sent
            output_dict['sent_labels'] = sent_labels_list
            output_dict['evd_possible_chains'] = evd_possible_chains
            output_dict['ans_sent_idxs'] = ans_sent_idxs
            output_dict['_id'] = ids

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, evidence_f1_socre = self._f1_metrics.get_metric(reset)
        ans_in_evd = self.evd_ans_metric.get_metric(reset)
        metrics = {
                'evd_p': p,
                'evd_r': r,
                'evd_f1': evidence_f1_socre,
                'ans_in_evd': ans_in_evd
                }
        for name, tracker in self._loss_trackers.items():
            metrics[name] = tracker.get_metric(reset).item()
        return metrics

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


@Seq2SeqEncoder.register("indep_bert_span_gate")
class SpanGate(Seq2SeqEncoder):
    def __init__(self, span_dim, gate_self_att):
        super().__init__()
        self.modeled_gate = nn.Sequential(
            nn.Linear(span_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self._gate_self_att = gate_self_att

    def forward(self,
                spans_tensor: torch.FloatTensor,
                spans_mask: torch.FloatTensor,
                self_att_layer: Seq2SeqEncoder,
                sent_encoder: Seq2SeqEncoder):

        print("spans_tensor", spans_tensor.shape)
        batch_size, num_spans, max_batch_span_width = spans_mask.size()
        # Shape: (batch_size, num_spans, embedding_dim)
        max_pooled_span_emb = spans_tensor[:, :, 0, :]

        # shape: (batch_size, num_spans)
        max_pooled_span_mask = (torch.sum(spans_mask, dim=-1) >= 1).float()

        if self._gate_self_att:
            # self attention on spans representation
            # shape: (batch_size, num_spans, embedding_dim)
            max_pooled_span_emb = sent_encoder(max_pooled_span_emb, max_pooled_span_mask)
            # shape: (batch_size, num_spans, embedding_dim)
            att_max_pooled_span_emb, _, att_score = self_att_layer(max_pooled_span_emb, max_pooled_span_mask)
            max_pooled_span_emb = max_pooled_span_emb + att_max_pooled_span_emb
        else:
            att_score = None

        # Shape: (batch_size * num_spans, embedding_dim)
        max_pooled_span_emb = max_pooled_span_emb.view(batch_size*num_spans, spans_tensor.size(3))

        # Shape: (batch_size * num_spans, 2)
        gate_logit = self.modeled_gate(max_pooled_span_emb)
        gate_prob = F.softmax(gate_logit, dim=-1)
        gate_prob = torch.chunk(gate_prob, 2, dim=-1)[1].view(batch_size * num_spans, -1)

        dumb_gate = torch.full_like(gate_prob.view(batch_size * num_spans, -1).float(), 0.3)
        new_gate = torch.cat([dumb_gate, gate_prob], dim=-1)

        return gate_logit, gate_prob, new_gate, max_pooled_span_mask, att_score
