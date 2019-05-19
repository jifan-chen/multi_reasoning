from overrides import overrides
from allennlp.common.util import JsonDict
import json, pickle
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset_readers import MultiprocessDatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.tools import squad_eval
import numpy as np
import torch
import time
from my_library.metrics import AttF1Measure



@Predictor.register('coref_visualizer')
class CorefVisualizer(Predictor):
    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        """
        Override the original init function to load the coref result to memory for demo
        Model and dataset_reader are just treated as dummy arguments here, so the model path can be arbitrary.
        """
        self._model = model
        if type(dataset_reader) == MultiprocessDatasetReader:
            self._dataset_reader = dataset_reader.reader
        else:
            self._dataset_reader = dataset_reader
        with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/train_coref.json', 'r') as f:
            train = json.load(f)
        #with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor_coref.json', 'r') as f:
        #    dev = json.load(f)
        with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/train_e2e_coref.pkl', 'rb') as f:
            e2e_coref = pickle.load(f)
        with open('/scratch/cluster/jfchen/jason/multihopQA/hotpot/train_spacy_coref.pkl', 'rb') as f:
            spacy_coref = pickle.load(f)
        self.demo_dataset = {'train': [train, e2e_coref, spacy_coref],}
                             #'dev': dev}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        """
        Should not be used.
        """
        raise Exception("Should not be used")

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Override this function for demo
        Expects JSON object as ``{"dataset": d,
                                  "instance_idx": idx}``
        """
        start_time = time.time()
        dataset, e2e_coref_dataset, spacy_coref_dataset = self.demo_dataset[inputs['dataset']]
        idx = int(inputs['instance_idx']) % len(dataset)
        hotpot_instance = dataset[idx]
        e2e_coref = e2e_coref_dataset[idx]
        spacy_coref = spacy_coref_dataset[idx]
        return {"doc":              e2e_coref['coref_info']['document'],
                "attns":            None,
                "qc_scores":        None,
                "qc_scores_sp":     None,
                "pred_sent_labels": None,
                "pred_sent_probs":  None,
                "evd_measure":      None,
                "evd_attns":        None,
                "question":         hotpot_instance['question'],#" ".join(question_tokens),
                "question_tokens":  "None",#question_tokens,
                "answer":           "None",#" ".join(answer_texts),
                "predict":          "None",
                "f1":               0.,
                "sent_spans":       None,
                "sent_labels":      None,
                "coref_clusters":   {'spacy coref clusters': spacy_coref['coref_info'],
                                     'e2e coref clusters': e2e_coref['coref_info'],
                                     'allen coref clusters': {'document': spacy_coref['coref_info']['document'],
                                                              'clusters': hotpot_instance['coref_clusters']}}
               }

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
