import json, pickle
import argparse
import numpy as np
from tqdm import tqdm
import random
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from typing import List
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter



class AllenCorefPredictor:
    def __init__(self, path):
        archive = load_archive(path, cuda_device=0)
        self.predictor = Predictor.from_archive(archive)

    def predict_raw(self, document: str) -> JsonDict:
        return self.predictor.predict(document=document)

    def predict_batch_raw(self, documents: List[str]) -> List[JsonDict]:
        '''
        outputs = [out]
        out = {"document": [tokenised document text],
               "clusters":
                   [
                       [
                           [start_index, end_index],
                           [start_index, end_index]
                       ],
                       [
                           [start_index, end_index],
                           [start_index, end_index],
                           [start_index, end_index],
                       ],
                       ....
                   ]
              }
        '''
        batch_json = [{'document': doc} for doc in documents]
        return self.predictor.predict_batch_json(batch_json)

    def predict_tokenized_sents(self, sents_tokens: List[List[str]]) -> JsonDict:
        instance = self.predictor._dataset_reader.text_to_instance(sents_tokens)
        return self.predictor.predict_instance(instance)

    def predict_batch_tokenized_sents(self, sents_tokens_list: List[List[List[str]]]) -> List[JsonDict]:
        instances = [self.predictor._dataset_reader.text_to_instance(sents_tokens) for sents_tokens in sents_tokens_list]
        return self.predictor.predict_batch_instance(instances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label golden dependency heads map for hotpot dataset')
    parser.add_argument('path', help='path to hotpot dataset')
    #parser.add_argument('output', help='path to dep-labeled hotpot dataset')
    parser.add_argument('coref_output', help='path to result of coreference resolution')
    parser.add_argument('--num', type=int, help='number of data to evaluate', default=-1)
    #parser.add_argument('--draw', action='store_true', help='draw dep parsing tree', default=False)
    args = parser.parse_args()

    predictor = AllenCorefPredictor(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
    tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    with open(args.path, 'r') as f:
        data = json.load(f)
    if not args.num == -1:
        data = data[:args.num]

    # Label the coref clusters for each instance.
    # We store the full coref result from allennlp coref predictor
    # The structure is similar to hotpot dataset
    # which is a list in which each instance is a dictionary
    # {'_id':       The instance id that the ``coref_info`` is mapped to
    #  'coref_info':The coref results of the instance that has id ``_id`` in hotpot dataset,
    #               which is obtained by seeing all paragraphs of a single document
    #                   
    #  'pos':       The pos of each sentence.
    #               A list of element [title: str,
    #                                  pos: List[List[str]] -> a list of coarse-grained pos for sentences in 
    #                                                          paragraph ``title``
    #                                  tag: List[List[str]] -> a list of fine-grained pos for sentences in 
    #                                                          paragraph ``title``
    #                                 ]
    # }
    coref_results = []
    for d in tqdm(data):
        doc_sents_tokens = []
        poses = []
        #tags = []
        for title, para in d['context']:
            #para_poses = [title, []]
            para_poses = [title, [], []]
            #para_tags = [title, []]
            for sent in para:
                sent_toks = tokenizer.split_words(sent)
                doc_sents_tokens.append([tok.text for tok in sent_toks])
                para_poses[1].append([tok.pos_ for tok in sent_toks])
                #para_tags[1].append([tok.tag_ for tok in sent_toks])
                para_poses[2].append([tok.tag_ for tok in sent_toks])
            poses.append(para_poses)
            #tags.append(para_tags)
        if doc_sents_tokens:
            coref_output = predictor.predict_tokenized_sents(doc_sents_tokens)
        else:
            coref_output = None
        coref_dict = {'_id': d['_id'],
                      'coref_info': coref_output,
                      'pos': poses,}
                      #'tag': tags}
        coref_results.append(coref_dict)

    with open(args.coref_output, 'wb') as f:
        pickle.dump(coref_results, f)
