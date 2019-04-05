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
import spacy



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


def align_tokens(tar_toks, ref_toks):
    t = 0
    r = 0
    t_char_pos = 0
    r_char_pos = 0
    res = [-1]*len(tar_toks)
    while t < len(tar_toks):
        if r < len(ref_toks):
            t_tok_s, t_tok_e = t_char_pos, t_char_pos + len(tar_toks[t])
            r_tok_s, r_tok_e = r_char_pos, r_char_pos + len(ref_toks[r])
            if (r_tok_s <= t_tok_e) and (t_tok_s <= r_tok_e):
                if t_tok_s == r_tok_s and t_tok_e == r_tok_e:
                    res[t] = r
                r += 1
                t += 1
                t_char_pos = t_tok_e + 1
                r_char_pos = r_tok_e + 1
            elif r_tok_s > t_tok_e:
                t += 1
                t_char_pos = t_tok_e + 1
            elif t_tok_s > r_tok_e:
                r += 1
                r_char_pos = r_tok_e + 1
        else:
            break
    return res


def transform_cluster(cluster, alignment):
    transformed_cluster = []
    for span in cluster.mentions:
        for t_idx in range(span.start, span.end):
            if alignment[t_idx] < 0:
                break
        else:
            transformed_cluster.append((alignment[span.start], alignment[span.end-1]))
    if len(transformed_cluster) > 1:
        return transformed_cluster
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label golden dependency heads map for hotpot dataset')
    parser.add_argument('path', help='path to hotpot dataset')
    #parser.add_argument('output', help='path to dep-labeled hotpot dataset')
    parser.add_argument('coref_output', help='path to result of coreference resolution')
    parser.add_argument('--model', default='allen', help='the coref model to use', choices=['allen', 'spacy'])
    parser.add_argument('--num', type=int, help='number of data to evaluate', default=-1)
    #parser.add_argument('--draw', action='store_true', help='draw dep parsing tree', default=False)
    args = parser.parse_args()

    tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)
    if args.model == 'allen':
        predictor = AllenCorefPredictor(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
    elif args.model == 'spacy':
        predictor = spacy.load('en_coref_lg')

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
    # }
    coref_results = []
    for d in tqdm(data):
        doc_sents_tokens = []
        for title, para in d['context']:
            for sent in para:
                sent_toks = tokenizer.split_words(sent)
                doc_sents_tokens.append([tok.text for tok in sent_toks])
        if args.model == 'allen':
            if doc_sents_tokens:
                coref_output = predictor.predict_tokenized_sents(doc_sents_tokens)
            else:
                coref_output = None
        elif args.model == 'spacy':
            if doc_sents_tokens:
                flatten_doc_tokens = [tok for sent in doc_sents_tokens for tok in sent]
                doc_text = " ".join(flatten_doc_tokens)
                output = predictor(doc_text)
                if not "".join(flatten_doc_tokens) == "".join([tok.text for tok in output]):
                    print(flatten_doc_tokens)
                    print([tok.text for tok in output])
                    exit()
                alignment = align_tokens(output, flatten_doc_tokens)
                if output._.coref_clusters:
                    clusters = []
                    for c in output._.coref_clusters:
                        c = transform_cluster(c, alignment)
                        if c:
                            clusters.append(c)
                else:
                    clusters = []
                coref_output = {'document': flatten_doc_tokens,
                                'clusters': clusters}
            else:
                coref_output = None
        coref_dict = {'_id': d['_id'],
                      'coref_info': coref_output}
        coref_results.append(coref_dict)

    with open(args.coref_output, 'wb') as f:
        pickle.dump(coref_results, f)
