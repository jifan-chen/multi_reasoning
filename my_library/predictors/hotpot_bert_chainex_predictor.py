from typing import List, Iterator
from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
import json
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset_readers import MultiprocessDatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
import numpy as np
import torch


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
        #exit()
        pos = None
        for i, c in enumerate(chain):
            if c < 0:
                pos = i
        chain = chain[:pos]
    return chain


@Predictor.register('hotpot_bert_chainex_predictor')
class HotpotPredictor(Predictor):
    @overrides
    def _json_to_instance(self, hotpot_dict_instance: JsonDict) -> Instance:
        print(type(hotpot_dict_instance), len(hotpot_dict_instance))
        if type(self._dataset_reader) == MultiprocessDatasetReader:
            processed_instance = self._dataset_reader.reader.process_raw_instance(hotpot_dict_instance)
        else:
            processed_instance = self._dataset_reader.process_raw_instance(hotpot_dict_instance)
        instance = self._dataset_reader.text_to_instance(*processed_instance)
        return instance

    def process_output(self, output: JsonDict) -> JsonDict:
        pred_sent_orders = output.get('pred_sent_orders', None)
        num_sents = len(output['sent_labels']) # for removing padding
        if not pred_sent_orders is None:
            pred_chains = [order2chain(order) for order in pred_sent_orders]
            pred_chains = [ch for ch in pred_chains if all(c < num_sents for c in ch)]
            assert len(pred_chains) > 0, repr([order2chain(order) for order in pred_sent_orders]) + '\n' + 'num sents: %d' % num_sents + '\n%s' % output['_id']
        else:
            # get pred evdiences from sentences with top k ``gate_prob``
            gate_probs = output['gate_probs'][:num_sents]
            pred_chains = [[i] for i in sorted(range(num_sents), key=lambda x: gate_probs[x], reverse=True)[:10]]
        return {#'answer_texts': output['answer_texts'],
                #'best_span_str': output.get('best_span_str', None),
                #'best_span': output.get('best_span', None),
                'pred_sent_labels': output.get('pred_sent_labels', None),
                'pred_sent_orders': output.get('pred_sent_orders', None),
                'pred_chains': pred_chains,
                'possible_chain': output.get('evd_possible_chains', None),
                'question_tokens': output['question_tokens'],
                'passage_sent_tokens': output['passage_sent_tokens'],
                #'token_spans_sp': output['token_spans_sp'],
                #'token_spans_sent': output['token_spans_sent'],
                'sent_labels': output['sent_labels'],
                'ans_sent_idxs': output.get('ans_sent_idxs', None),
                '_id': output['_id']}

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs = sanitize(outputs)
        return self.process_output(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        outputs = sanitize(outputs)
        return [self.process_output(o) for o in outputs]

    def predict(self, hotpot_dict_instance: JsonDict) -> JsonDict:
        """
        Expects JSON that has the same format of instances in Hotpot dataset
        """
        return self.predict_json(hotpot_dict_instance)
