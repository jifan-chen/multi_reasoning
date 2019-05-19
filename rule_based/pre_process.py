import json
import spacy
import argparse
from collections import defaultdict
from rouge import Rouge
from allennlp.predictors.predictor import Predictor

rouge = Rouge()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger'])


def entity_extraction(sentence):
    entities = list(nlp(sentence).ents)
    return entities


def entity_extraction_(words, tags):
    in_entity = False
    entities = []
    phrase = []
    for w, t in zip(words, tags):
        if t != 'O':
            phrase.append(w)
            in_entity = True
        elif t == 'O' and in_entity:
            in_entity = False
            entities.append(" ".join(phrase).lower())
            phrase = []
    return entities


def build_entity_links(data):
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    # predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")

    output = []
    for article in data[:10]:
        article_id = article['_id']
        paragraphs = article['context']
        question_text = article['question'].strip().replace("\n", "")
        print('original:', question_text)
        predicted = predictor.predict(sentence=question_text)
        entities_in_question = entity_extraction_(predicted['words'], predicted['tags'])
        entities_each_sent = []
        answer_text = article['answer'].strip().replace("\n", "")
        sp_set = set(list(map(tuple, article['supporting_facts'])))
        sp_sents = []
        meta_sents= []
        entity2sent = defaultdict(list)
        sent_pair2entity = defaultdict(list)
        global_sent_id = 0
        start_point = set()
        max_overlap_id = 0
        max_rouge = 0
        answer_labels = []
        paras = []
        print('-*' * 30)

        for para in paragraphs:
            para_sents = []
            cur_title, cur_para = para[0], para[1]
            for sent_id, sent in enumerate(cur_para):
                predicted = predictor.predict(sentence=sent)
                entities = entity_extraction_(predicted['words'], predicted['tags'])
                entities_each_sent.append([entities, global_sent_id])
                print(sent)
                print(entities)
                if (cur_title, sent_id) in sp_set:
                    sp_sents.append(global_sent_id)
                    if answer_text in sent:
                        answer_labels.append(global_sent_id)
                para_sents.append(global_sent_id)
                meta_sents.append(sent)

                # if len(set(entities_in_question).intersection(set(entities))) > 0:
                #     start_point.append(global_sent_id)
                    # print('entities in question:', set(entities_in_question))
                    # print('entities of current sent:', set(entities))
                    # print(global_sent_id, sent)
                    # print(set(entities_in_question).intersection(set(entities)))

                if len(sent) > len(question_text) / 2:
                    rouge_score = rouge.get_scores(sent, question_text)
                    rf1 = rouge_score[0]['rouge-1']['f']
                    if rf1 > max_rouge:
                        max_rouge = rf1
                        max_overlap_id = global_sent_id

                for e in entities:
                    if global_sent_id not in entity2sent[e]:
                        entity2sent[e].append(global_sent_id)
                global_sent_id += 1

            paras.append(para_sents)
        # print('entities in question:', set(entities_in_question))
        print(entities_each_sent)
        for sent_entities, sent_id in entities_each_sent:
            # print('entities of current sent:', set(sent_entities))
            shared = set(entities_in_question).intersection(set(sent_entities))
            if len(shared) > 0:

                # for e in shared:
                #     if len(entity2sent[e]) < 4:
                start_point.add(sent_id)

        # if len(start_point) == 0:
        start_point.add(max_overlap_id)

        for k in entity2sent.keys():
            sents = entity2sent[k]
            # make sure it is not a common entity
            if len(sents) < 3:
                for i in range(len(sents)):
                    for j in range(i+1, len(sents)):
                        # print(sents[i], sents[j], k)
                        # print(sents)
                        # input()
                        sent_pair2entity[" ".join((str(sents[i]), str(sents[j])))].append(k)
        print(sent_pair2entity)
        print('start_points:', start_point)
        print('answer_labels:', answer_labels)
        print(paras)
        dict_wrap = {
            "pair2entity": sent_pair2entity,
            "answer_labels": answer_labels,
            "start_points": list(start_point),
            "para_sents": paras,
            "sp_sents": sp_sents,
            "meta_sents": meta_sents,
            "_id": article_id
        }
        output.append(dict_wrap)
    return output


def write_json(data, output_path):
    json.dump(data, open(output_path, 'w'), indent=4, sort_keys=True)


def calculate_yes_no(data):
    count = 0
    for article in data:
        answer_text = article['answer'].strip().replace("\n", "")
        if answer_text == 'yes' or answer_text == 'no':
            count += 1
    print(count / float(len(data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('--data_path', type=str, help='path to the input data file')
    parser.add_argument('--output_path', type=str, help='path to the output file', default="")

    args = parser.parse_args()
    json_data = json.load(open(args.data_path))
    output_data = build_entity_links(json_data)
    write_json(output_data, args.output_path)
