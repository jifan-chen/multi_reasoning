from overrides import overrides
from allennlp.common.util import JsonDict
import json
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('hotpot_predictor')
class HotpotPredictor(Predictor):
    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        """
        Override the original init function to load the dataset to memory for demo
        """
        self._model = model
        self._dataset_reader = dataset_reader
        with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/test/test_10000_coref.json', 'r') as f:
            self.demo_dataset = json.load(f)

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
        Expects JSON object as ``{"instance_idx" : idx}``
        """
        idx = inputs['instance_idx'] % len(self.demo_dataset)
        hotpot_instance = self.demo_dataset[idx]
        outputs = self.predict(hotpot_instance)
        return " ".join(outputs['passage_tokens'])

    def _predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Serve as the substitute for the original ``predict_json``
        """
        instance = self._json_to_instance(inputs)
        return {"passage": self.predict_instance(instance)}

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
