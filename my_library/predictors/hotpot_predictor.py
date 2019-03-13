from overrides import overrides
from allennlp.common.util import JsonDict
import json
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.tools import squad_eval
import numpy as np


def find_att_toks(scores, mask, th, row_offset, conn_type):
    scores = scores * mask
    accept = scores >= th
    accept_row = np.sum(accept, axis=1) >= 1
    accept_scores = scores * accept
    row_idx = (np.arange(scores.shape[0])[accept_row] + row_offset).tolist()
    row_accept_scores = accept_scores[accept_row, :].tolist()
    return list(map(lambda x: {'target': "", 'pos': x[0], 'scores': x[1], 'type': conn_type},
                    zip(row_idx, row_accept_scores)))


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


def analyze_att(att_scores, labels, metric, row_spans, col_spans, num_att_heads, TH):
    self_mask = 1 - np.identity(att_scores.shape[1])
    all_att_toks = [{'T': [], 'F': []} for h_idx in range(num_att_heads)]
    for (r_s, r_e), (c_s, c_e) in zip(row_spans, col_spans):
        att_scores_sp = att_scores[:, r_s:r_e+1, c_s:c_e+1]
        labels_sp = labels[r_s:r_e+1, c_s:c_e+1]
        self_mask_sp = self_mask[r_s:r_e+1, c_s:c_e+1]
        for h_idx in range(num_att_heads):
            att_toks = find_att_toks(att_scores_sp[h_idx], labels_sp*self_mask_sp, TH, r_s, 'T')
            all_att_toks[h_idx]['T'].extend(att_toks)
            att_toks = find_att_toks(att_scores_sp[h_idx], (~labels_sp)*self_mask_sp, TH, r_s, 'F')
            all_att_toks[h_idx]['F'].extend(att_toks)
    return all_att_toks


def get_coref_map(coref_clusters, seq_len, passage_tokens):
    m = np.zeros((seq_len, seq_len))
    for c in coref_clusters:
        for i in range(0, len(c)-1):
            for j in range(i+1, len(c)):
                i_s, i_e = c[i]
                j_s, j_e = c[j]
                if not " ".join(passage_tokens[i_s:i_e+1]).lower() == " ".join(passage_tokens[j_s:j_e+1]).lower():
                    m[i_s:i_e+1, j_s:j_e+1] = 1
                    m[j_s:j_e+1, i_s:i_e+1] = 1
    return m


@Predictor.register('hotpot_predictor')
class HotpotPredictor(Predictor):
    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        """
        Override the original init function to load the dataset to memory for demo
        """
        self._model = model
        self._dataset_reader = dataset_reader
        # with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/test/test_10000_coref.json', 'r') as f:
        #     train = json.load(f)
        # with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor_coref.json', 'r') as f:
        #     dev = json.load(f)
        # self.demo_dataset = {'train': train,
        #                      'dev': dev}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like
        ``{"question_text":     "...",
           "passage_text":      "...",
           "char_spans":        [(ans_start, ans_end), ...],
           "char_spans_sp":     [(sup_start, sup_end), ...],
           "char_spans_sent":   [(sent_start, sent_end), ...],
           "sent_labels":       [0, 1, 0, ...],
           "answer_texts":      ["..."],
           "passage_tokens":    [Token, ...],
           "passage_offsets":   [(token_start, token_end), ...],
           "passage_dep_heads": [(child_tok_idx, parent_tok_idx), ...]),
           "coref_clusters":    [[[start, end], [start, end], ...], ...],
           "article_id":        "..."
            }``.
        """
        return self._dataset_reader.text_to_instance(**json_dict)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Override this function for demo
        Expects JSON object as ``{"dataset": d,
                                  "th": th,
                                  "instance_idx": idx}``
        """
        dataset = self.demo_dataset[inputs['dataset']]
        TH = float(inputs['th'])
        idx = int(inputs['instance_idx']) % len(dataset)
        hotpot_instance = dataset[idx]
        output = self.predict(hotpot_instance)
        loss                = output['loss']
        passage_tokens      = output['passage_tokens']
        question_tokens     = output['question_tokens']
        token_spans_sp      = output['token_spans_sp']
        token_spans_sent    = output['token_spans_sent']
        sent_labels         = output['sent_labels']
        coref_clusters      = output['coref_clusters']
        self_att_scores     = output['self_attention_score']
        answer_texts        = output['answer_texts']
        best_span_str       = output['best_span_str']
        article_id          = output['_id']
        self_att_scores = np.array(self_att_scores)
        em, f1 = calc_em_and_f1(best_span_str, answer_texts)
        num_att_heads = self_att_scores.shape[0]
	# coref res
        coref_map = get_coref_map(coref_clusters, len(passage_tokens), passage_tokens)
        coref_map = coref_map.astype(bool)
        assert self_att_scores.shape == (num_att_heads, len(passage_tokens), len(passage_tokens))
        assert np.allclose(np.sum(self_att_scores, axis=2), 1.)
        assert len(sent_labels) == len(token_spans_sent)
        for i, (s, e) in enumerate(token_spans_sent):
            if not i == 0:
                assert s == token_spans_sent[i-1][1] + 1
            if not i == len(token_spans_sent) - 1:
                assert e == token_spans_sent[i+1][0] - 1
        col_spans = [(0, len(passage_tokens)-1)]*len(token_spans_sp)
        att_toks = analyze_att(self_att_scores, coref_map, f1, token_spans_sp, col_spans, num_att_heads, TH)
        # find att tokens
        heads_doc_res = []
        for h_idx in range(num_att_heads):
            doc_res = []
            for tok_dict in att_toks[h_idx]['T']:
                assert len(tok_dict['target']) == 0
                tok_dict['target'] = passage_tokens[tok_dict['pos']]
                doc_res.append(tok_dict)
            for tok_dict in att_toks[h_idx]['F']:
                assert len(tok_dict['target']) == 0
                tok_dict['target'] = passage_tokens[tok_dict['pos']]
                doc_res.append(tok_dict)
            heads_doc_res.append(doc_res)
        return {"doc": passage_tokens,
                "attns": heads_doc_res,
                "question": " ".join(question_tokens),
                "answer": " ".join(answer_texts),
                "predict": best_span_str,
                "f1": f1,
                "sent_spans": [list(sp) for sp in token_spans_sent],
                "sent_labels": sent_labels}

    def _predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Serve as the substitute for the original ``predict_json``
        """
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict(self, hotpot_instance: JsonDict) -> JsonDict:
        """
        Expects JSON that has the same format of instances in Hotpot dataset
        """
        processed_instance = self._dataset_reader.process_raw_instance(hotpot_instance)
        json_dict = {"question_text":       processed_instance[0],
                     "passage_text":        processed_instance[1],
                     "char_spans":          processed_instance[2],
                     "char_spans_sp":       processed_instance[3],
                     "char_spans_sent":     processed_instance[4],
                     "sent_labels":         processed_instance[5],
                     "answer_texts":        processed_instance[6],
                     "passage_tokens":      processed_instance[7],
                     "passage_offsets":     processed_instance[8],
                     "passage_dep_heads":   processed_instance[9],
                     "coref_clusters":      processed_instance[10],
                     "article_id":          processed_instance[11]
                     }
        return self._predict_json(json_dict)