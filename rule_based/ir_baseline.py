import multiprocessing
from textwrap import dedent
import json
import numpy as np
import time
from nltk import word_tokenize
from nltk import sent_tokenize
# from multiprocessing import Pool
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch()
TYPE = 'sentence'

mapping = '''
    {  
    "settings" : {
            "number_of_shards" : 1,
            "number_of_replicas" : 0  
        },
    "mappings": {
        "sentence": {
          "dynamic": "false",
          "properties": {
            "docId": {
              "type": "keyword"
            },
            "text": {
              "analyzer": "whitespace",
              "type": "text",
              "fields": {
                "raw": {
                  "type": "keyword"
                }
              }
            },
            "tags": {
              "type": "keyword"
            }
          }
        }
      }
    }'''


def extract_evidences_by_IR(index_name, query, answer, all_sents, max_hits_retrived=10):
    print(query)
    print(answer)

    constructed_query = {"from": 0, "size": max_hits_retrived + 1,
                         "query": {
                             "bool": {
                                 "must": [
                                     {"match": {
                                         "text": query
                                     }}
                                 ]
                             }
                         }}
    result = es.search(index=index_name, body=constructed_query)

    evidences = []
    evidence_idxs = []
    print(all_sents)
    for hit in result['hits']['hits']:
        print("hit:", hit["_source"]["text"])
        # input()
        evidences.append(hit["_source"]["text"])
        evidence_idxs.append([all_sents.index(hit["_source"]["text"])])

    print('evidence set:', evidences)
    print('evidence idxs:', evidence_idxs)

    return evidences, evidence_idxs


def build_index(passage,index_name):

    def yield_document(passage):
        sentence_id = 0
        for p in passage:
            doc = {
                '_op_type': 'create',
                '_index': index_name,
                '_type': TYPE,
                '_id': sentence_id,
                '_source': {'text': p.strip()}
            }
            sentence_id += 1
            yield (doc)

    try:
        es.indices.delete(index_name)
        res = es.indices.create(index=index_name, ignore=400, body=mapping)
        res = bulk(es, yield_document(passage))
    except Exception:
        res = es.indices.create(index=index_name, ignore=400, body=mapping)
        res = bulk(es, yield_document(passage))


def process_chunk(obj):
    """Replace this with your own function
    that processes data one line at a
    time"""
    if obj is not None:
        passage = word_tokenize(' '.join(obj['supports']).lower())
        psg_sts = sent_tokenize(' '.join(passage))

        build_index(psg_sts,obj['id'].lower())

        return True
#
# def grouper(n, iterable, padvalue=None):
#     """grouper(3, 'abcdefg', 'x') -->
#     ('a','b','c'), ('d','e','f'), ('g','x','x')"""
#
#     return izip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def extract_wikihop(path):
    json_data = json.load(open(path, 'r'))[:100]
    batch = []
    sample_count = 0
    not_found_count = 0
    batch_size = 100

    for i, obj in enumerate(json_data):
        pragraphs = obj['passage']
        question_text = obj['question'].strip().replace("\n", "")
        answer_text = obj['answer'].strip().replace("\n", "")
        all_sents = []
        sent_labels = []
        answer_labels = []
        global_id = 0
        for para in pragraphs:
            for sent_id, sent in enumerate(para):
                global_id += 1
                all_sents.append(sent.strip())
                sent_labels.append(0)

                if answer_text in sent:
                    answer_labels.append(global_id)

        idx_name = 'sample' + str(sample_count)
        build_index(all_sents, idx_name)
        batch.append((obj, all_sents, sent_labels, answer_labels, idx_name))
        sample_count += 1

        if sample_count == batch_size or i == len(json_data) - 1:
            time.sleep(5)
            for o, sents, sent_label, answer_label, n in batch:
                print('extracting evidences:', n, o['question'])
                evidence_set, evidence_idxs = \
                    extract_evidences_by_IR(n, o['question'], o['answer'], sents)

                o['pred_chains'] = evidence_idxs
                o['sent_labels'] = sent_label
                o['ans_sent_idxs'] = answer_label
            batch = []
            sample_count = 0

    print('no match count:', not_found_count)
    json.dump(json_data, open('ir_based_wikihop.json', 'w'), indent=4, sort_keys=True)


def extract_hotpot(path):
    json_data = json.load(open(path, 'r'))[:100]
    batch = []
    sample_count = 0
    not_found_count = 0
    batch_size = 100

    for i, obj in enumerate(json_data):
        pragraphs = obj['context']
        question_text = obj['question'].strip().replace("\n", "")
        answer_text = obj['answer'].strip().replace("\n", "")
        sp_set = set(list(map(tuple, obj['supporting_facts'])))
        all_sents = []
        sent_labels = []
        answer_labels = []
        global_id = 0
        for para in pragraphs:
            cur_title, cur_para = para[0], para[1]
            for sent_id, sent in enumerate(cur_para):
                global_id += 1
                all_sents.append(sent.strip())
                if (cur_title, sent_id) in sp_set:
                    sent_labels.append(1)
                else:
                    sent_labels.append(0)

                if answer_text in sent:
                    answer_labels.append(global_id)

        idx_name = 'sample' + str(sample_count)
        build_index(all_sents, idx_name)
        batch.append((obj, all_sents, sent_labels, answer_labels, idx_name))
        sample_count += 1

        if sample_count == batch_size or i == len(json_data) - 1:
            time.sleep(5)
            for o, sents, sent_label, answer_label, n in batch:
                print('extracting evidences:', n, o['question'])
                evidence_set, evidence_idxs = \
                    extract_evidences_by_IR(n, o['question'], o['answer'], sents)

                o['pred_chains'] = evidence_idxs
                o['sent_labels'] = sent_label
                o['ans_sent_idxs'] = answer_label
            batch = []
            sample_count = 0

    print('no match count:', not_found_count)
    json.dump(json_data, open('ir_based_hotpot.json', 'w'), indent=4, sort_keys=True)


def main():
    extract_wikihop('/backup2/jfchen/data/wikihop/dev_chain.json')
    # extract_hotpot('/backup2/jfchen/data/hotpot/dev/dev_distractor_chain.json')


if __name__ == '__main__':
    main()

