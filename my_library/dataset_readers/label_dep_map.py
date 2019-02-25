import json, pickle
import argparse
import numpy as np
from tqdm import tqdm
import random
from spacy import displacy
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import sanitize
from allennlp.models.archival import load_archive


def get_display_form(data, type):
    ret = {'words': [{'text': 'ROOT', 'tag': 'ROOT'}],
           'arcs': []}
    if type == 'allen':
        for w, pos in zip(data['words'], data['pos']):
            ret['words'].append({'text': w, 'tag': pos})
        for i, (head, dep) in enumerate(zip(data['predicted_heads'], data['predicted_dependencies'])):
            ret['arcs'].append({'start': head if head < i+1 else i+1,
                                'end': head if head > i+1 else i+1,
                                'label': dep,
                                'dir': ['left', 'right'][head<i+1]})
    elif type == 'conllx':
        for w, pos in zip(data['words'], data['pos']):
            ret['words'].append({'text': w, 'tag': pos})
        for i, (head, dep) in enumerate(zip(data['heads'], data['dependencies'])):
            ret['arcs'].append({'start': head if head < i+1 else i+1,
                                'end': head if head > i+1 else i+1,
                                'label': dep,
                                'dir': ['left', 'right'][head<i+1]})
    return ret


def gen_dep_map(predicted_heads):
    n = len(predicted_heads)
    map = np.zeros((n, n), dtype='int')
    valid_heads = [h-1 for h in predicted_heads if h > 0]
    valid_childs = [i for i, h in enumerate(predicted_heads) if h > 0]
    map[valid_heads+valid_childs, valid_childs+valid_heads] = 1
    return map.tolist()


class AllenDepParser:
    def __init__(self, path):
        archive = load_archive(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
            cuda_device=0)
        self.predictor = Predictor.from_archive(archive)

    def predict_raw(self, sentence):
        instance = self.predictor._json_to_instance({'sentence': sentence})
        output = self.predictor._model.forward_on_instance(instance)
        return sanitize(output)

    def predict_batch_raw(self, sentences):
        '''
        outputs = [out]
        out = {'words': ...
               'pos': ...
               'predicted_heads': ...
               'predicted_dependencies': ...}
        '''
        batch_json = [{'sentence': sent} for sent in sentences]
        instances = self.predictor._batch_json_to_instances(batch_json)
        outputs = self.predictor._model.forward_on_instances(instances)
        return sanitize(outputs)

    def predict_tokens(self, tokens):
        instance = self.predictor._dataset_reader.text_to_instance(tokens[0], tokens[1])
        output = self.predictor._model.forward_on_instance(instance)
        return sanitize(output)

    def predict_batch_tokens(self, tokens_list):
        instances = [self.predictor._dataset_reader.text_to_instance(toks, tags) for toks, tags in tokens_list]
        outputs = self.predictor._model.forward_on_instances(instances)
        return sanitize(outputs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label golden dependency heads map for hotpot dataset')
    parser.add_argument('path', help='path to hotpot dataset')
    parser.add_argument('output', help='path to dep-labeled hotpot dataset')
    parser.add_argument('dep_output', help='path to result of dependency parsing')
    parser.add_argument('--num', type=int, help='number of data to evaluate', default=-1)
    parser.add_argument('--draw', action='store_true', help='draw dep parsing tree', default=False)
    args = parser.parse_args()

    predictor = AllenDepParser(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    with open(args.path, 'r') as f:
        data = json.load(f)
    if not args.num == -1:
        data = data[:args.num]

    # Label the dependency parents for each sentence.
    # We insert a new key ``golden_head`` for each instance.
    # The structure of instance['golden_head'] is basically the same as instance['context'],
    # which is
    # [
    #    [
    #        title -> str,
    #        [head -> List[int], ...]
    #    ],
    #    .
    #    .
    #    .
    # ]
    # All head indices are 1-indexing.
    #
    # We also store the full dependency parsing result from allennlp parser
    # The structure is similar to hotpot dataset
    # which is a list in which each instance is a dictionary
    # {'_id':           The instance id that the ``parsing_info`` is mapped to
    #  'parsing_info':  The parsing results of the instance that has id ``_id`` in hotpot dataset.
    #                   A list of element [title-> str,
    #                                      results-> a list of parsing results for sentences in paragraph ``title``
    #                                     ]
    # }
    dep_results = []
    for d in tqdm(data):
        golden_heads = []
        parsing_info = []
        for title, para in d['context']:
            para_heads = [title, []]
            outputs = predictor.predict_batch_raw(para)
            for out in outputs:
                para_heads[1].append(out['predicted_heads'])
            golden_heads.append(para_heads)

            parsing_info.append([title, outputs])
        d['golden_head'] = golden_heads
        dep_dict = {'_id': d['_id'],
                    'parsing_info': parsing_info}
        dep_results.append(dep_dict)

    with open(args.output, 'w') as f:
        json.dump(data, f)
    with open(args.dep_output, 'wb') as f:
        pickle.dump(dep_results, f)

    # draw dep trees
    if args.draw:
        draw_idxs = random.sample(range(len(dep_results)), 5)
        draw_data = [get_display_form(out, 'allen')
                     for idx in draw_idxs
                        for t, outputs in dep_results[idx]['parsing_info']
                            for out in outputs
                     ]
        draw_idxs = random.sample(range(len(draw_data)), 10)
        draw_data = [draw_data[idx] for idx in draw_idxs]
        displacy.serve(draw_data, style='dep', manual=True)