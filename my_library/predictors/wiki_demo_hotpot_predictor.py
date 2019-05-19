from overrides import overrides
from allennlp.common.util import JsonDict
import json
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset_readers import MultiprocessDatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.tools import squad_eval
import numpy as np
import torch
import time
from my_library.metrics import AttF1Measure


def order2chain(order):
    chain = []
    for s_idx, o in enumerate(order):
        o = int(o)
        if o >= 0:
            if o >= len(chain):
                chain += [-1]*(o+1-len(chain))
            chain[o] = s_idx
    if not all([i >= 0 for i in chain]):
        print("order:", order)
        print("chain:", chain)
        exit()
    return chain


def chain_EM(pred_chain, rb_chain):
    return pred_chain == rb_chain


def find_att_toks(scores, labels, mask, th):
    scores = scores * mask
    accept = scores >= th
    accept_row = np.sum(accept, axis=1) >= 1
    accept_scores = scores * accept
    accept_labels = labels & accept
    row_idx = (np.arange(scores.shape[0])[accept_row]).tolist()
    row_accept_scores = accept_scores[accept_row, :].tolist()
    row_accept_labels = accept_labels[accept_row, :].tolist()
    return list(map(lambda x: {'target': "", 
                               'pos': x[0], 
                               'scores': x[1], 
                               'type': 'T' if sum(x[2]) > 0 else 'F', 
                               'labels': x[2]},
                    zip(row_idx, row_accept_scores, row_accept_labels)))


def calc_em_and_f1(best_span_string, answer_strings):
    exact_match = squad_eval.metric_max_over_ground_truths(
            squad_eval.exact_match_score,
            best_span_string,
            answer_strings)
    f1_score = squad_eval.metric_max_over_ground_truths(
            squad_eval.f1_score,
            best_span_string,
            answer_strings)
    return exact_match, f1_score


def calc_evd_f1(pred_labels, gold_labels):
    evd_metric = AttF1Measure(0.5) # We just use 0.5 as the TH since pred_labels should only contain 0 and 1
    T_P, N_P, N_T = evd_metric(torch.tensor(pred_labels).float(), torch.tensor(gold_labels).float())
    precision = float(T_P) / float(N_P + 1e-13)
    recall = float(T_P) / float(N_T + 1e-13)
    f1 = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1


def analyze_att(att_scores, labels, num_att_heads, TH):
    self_mask = 1 - np.identity(att_scores.shape[1])
    all_att_toks = [[] for h_idx in range(num_att_heads)]
    for h_idx in range(num_att_heads):
        att_toks = find_att_toks(att_scores[h_idx], labels, self_mask, TH)
        all_att_toks[h_idx].extend(att_toks)
    return all_att_toks


def get_coref_map(coref_clusters, seq_len, passage_tokens):
    m = np.zeros((seq_len, seq_len))
    for c in coref_clusters:
        for i in range(0, len(c)-1):
            for j in range(i+1, len(c)):
                i_s, i_e = c[i]
                j_s, j_e = c[j]
                '''
                if not " ".join(passage_tokens[i_s:i_e+1]).lower() == " ".join(passage_tokens[j_s:j_e+1]).lower():
                    m[i_s:i_e+1, j_s:j_e+1] = 1
                    m[j_s:j_e+1, i_s:i_e+1] = 1
                '''
                m[i_s:i_e+1, j_s:j_e+1] = 1
                m[j_s:j_e+1, i_s:i_e+1] = 1
    return m


@Predictor.register('wiki_demo_hotpot_predictor')
class WikiDemoHotpotPredictor(Predictor):
    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        """
        Override the original init function to load the dataset to memory for demo
        """
        self._model = model
        if type(dataset_reader) == MultiprocessDatasetReader:
            self._dataset_reader = dataset_reader.reader
        else:
            self._dataset_reader = dataset_reader
        '''
        with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/test/test_10000_chain.json', 'r') as f:
            train = json.load(f)
        '''
        with open('/scratch/cluster/jfchen/jason/multihopQA/wikihop/dev/dev_chain.json', 'r') as f:
            dev = json.load(f)
        '''
        with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor_chain_easy.json', 'r') as f:
            dev_easy = json.load(f)
        with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor_chain_hard.json', 'r') as f:
            dev_hard = json.load(f)
        '''
        #self.demo_dataset = {'train': train,
        self.demo_dataset = {'dev': dev,}
                             #'dev_easy': dev_easy,
                             #'dev_hard': dev_hard}

    @overrides
    def _json_to_instance(self, hotpot_dict_instance: JsonDict) -> Instance:
        processed_instance = self._dataset_reader.process_raw_instance(hotpot_dict_instance)
        instance = self._dataset_reader.text_to_instance(*processed_instance)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Override this function for demo
        Expects JSON object as ``{"dataset": d,
                                  "th": th,
                                  "instance_idx": idx}``
        """
        start_time = time.time()
        dataset = self.demo_dataset[inputs['dataset']]
        TH = float(inputs['th'])
        idx = int(inputs['instance_idx']) % len(dataset)
        hotpot_instance = dataset[idx]
        output = self.predict(hotpot_instance)
        print("pred:", time.time() - start_time)
        loss                = output['loss']
        passage_tokens      = output['passage_tokens']
        question_tokens     = output['question_tokens']
        token_spans_sp      = output['token_spans_sp']
        token_spans_sent    = output['token_spans_sent']
        sent_labels         = output['sent_labels']
        pred_sent_labels    = output.get('pred_sent_labels', None)
        coref_clusters      = output['coref_clusters']
        self_att_scores     = output.get('self_attention_score', None)
        evd_self_att_scores = output.get('evd_self_attention_score', None)
        qc_scores           = output['qc_score']
        qc_scores_sp        = output.get('qc_score_sp', None)
        gate_probs          = output.get('gate_probs', None)
        pred_sent_orders    = output.get('pred_sent_orders', None)
        answer_texts        = output['answer_texts']
        evd_possible_chains = output.get('evd_possible_chains', None)
        ans_sent_idxs       = output.get('ans_sent_idxs', None)
        best_span_str       = output['best_span_str']
        article_id          = output['_id']
        em, f1 = calc_em_and_f1(best_span_str, answer_texts)
        if not pred_sent_labels is None:
            pred_sent_labels = np.array(pred_sent_labels)
            evd_prec, evd_recl, evd_f1 = calc_evd_f1(pred_sent_labels, sent_labels)
        else:
            evd_prec, evd_recl, evd_f1 = None, None, None
        evd_measure = {'prec': evd_prec, 'recl': evd_recl, 'f1': evd_f1}
        if not evd_self_att_scores is None:
            evd_self_att_scores = np.transpose(evd_self_att_scores, (1, 2, 0))
	# coref res
        if not self_att_scores is None:
            self_att_scores = np.array(self_att_scores)
            if len(self_att_scores.shape) == 2:
                self_att_scores = self_att_scores[None, :, :]
            num_att_heads = self_att_scores.shape[0]
            coref_map = get_coref_map(coref_clusters, len(passage_tokens), passage_tokens)
            coref_map = coref_map.astype(bool)
            assert self_att_scores.shape == (num_att_heads, len(passage_tokens), len(passage_tokens))
            #assert np.allclose(np.sum(self_att_scores, axis=2), 1.)
            assert len(sent_labels) == len(token_spans_sent)
            att_toks = analyze_att(self_att_scores, coref_map, num_att_heads, TH)
            # find att tokens
            heads_doc_res = []
            for h_idx in range(num_att_heads):
                doc_res = []
                for tok_dict in att_toks[h_idx]:
                    assert len(tok_dict['target']) == 0
                    tok_dict['target'] = passage_tokens[tok_dict['pos']]
                    doc_res.append(tok_dict)
                heads_doc_res.append(doc_res)
        else:
            heads_doc_res = None
        if not pred_sent_orders is None:
            pred_chains = [order2chain(order) for order in pred_sent_orders]
            pred_chains_em = [chain_EM(chain, evd_possible_chains) for chain in pred_chains]
            # now just take the first one to visualize
            #pred_sent_orders = pred_sent_orders[0]
        print(coref_clusters)
        print(type(coref_clusters))
        coref_clusters = None
        print("fin:", time.time() - start_time)
        return {"doc":              passage_tokens,
                "attns":            heads_doc_res,
                "qc_scores":        qc_scores,
                "qc_scores_sp":     qc_scores_sp,
                "pred_sent_labels": (pred_sent_labels).astype("int").tolist() if not pred_sent_labels is None else None,
                "pred_sent_probs":  gate_probs,
                "pred_sent_orders": pred_sent_orders,
                "pred_chains":      pred_chains if not pred_sent_orders is None else None,
                "rb_chains":        evd_possible_chains,
                "evd_measure":      evd_measure,
                "evd_attns":        evd_self_att_scores.tolist() if not evd_self_att_scores is None else None,
                "question":         " ".join(question_tokens),
                "question_tokens":  question_tokens,
                "answer":           " ".join(answer_texts),
                "predict":          best_span_str,
                "f1":               f1,
                "chain_em":         pred_chains_em[0] if not pred_sent_orders is None else None,
                "topk_chain_em":    any(pred_chains_em) if not pred_sent_orders is None else None,
                "sent_spans":       [list(sp) for sp in token_spans_sent],
                "sent_labels":      sent_labels,
                "coref_clusters":   {'coref clusters': {'document': passage_tokens, 'clusters': coref_clusters}} if coref_clusters else None,
                "ans_sent_idxs":     ans_sent_idxs}

    def _predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Serve as the substitute for the original ``predict_json``
        """
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict(self, hotpot_dict_instance: JsonDict) -> JsonDict:
        """
        Expects JSON that has the same format of instances in Hotpot dataset
        """
        return self._predict_json(hotpot_dict_instance)
