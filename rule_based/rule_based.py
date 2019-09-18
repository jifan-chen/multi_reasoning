import json
import argparse
from tqdm import tqdm
from rouge import Rouge
from collections import defaultdict
rouge = Rouge()


def build_graph(sent_pairs):
    entity_graph = defaultdict(list)
    for pair in sent_pairs:
        # print(pair)
        entity_graph[int(pair[0])].append(int(pair[1]))
        entity_graph[int(pair[1])].append(int(pair[0]))
    return entity_graph


def construct_graph_edge_by_entities(entity2sent):
    sent_edges = set()
    for k in entity2sent.keys():
        sents = entity2sent[k]
        # make sure it is not a common entity
        if len(sents) < 5:
            for i in range(len(sents)):
                for j in range(i + 1, len(sents)):
                    # print(sents[i], sents[j], k)
                    # print(sents)
                    # input()
                    sent_edges.add((sents[i], sents[j]))
    return list(sent_edges)


def get_start_points_hotpot(question_entities, entities_each_sent, max_overlap_id):
    start_points = set()
    for sent_entities, sent_id in entities_each_sent:
        shared = set(question_entities).intersection(set(sent_entities))
        if len(shared) > 0:
            start_points.add(sent_id)
    start_points.add(max_overlap_id)
    return start_points


def preprocess_hotpot(article):
    article_id = article['_id']
    paragraphs = article['context']
    ner_paragraphs = article['ners']
    question_text = article['question'].strip().replace("\n", "")
    entities_in_question = article['question_entities']
    answer_text = article['answer'].strip().replace("\n", "")
    sp_set = set(list(map(tuple, article['supporting_facts'])))
    entities_each_sent = []
    sp_sents = []
    meta_sents = []
    answer_labels = []
    paras = []
    entity2sent = defaultdict(list)
    global_sent_id = 0
    max_overlap_id = 0
    max_rouge = 0

    for para, para_ner in zip(paragraphs, ner_paragraphs):
        para_sents = []
        cur_title, cur_para = para[0], para[1]
        ner_title, ner_para = para_ner[0], para_ner[1]
        for sent_id, (sent, sent_ner) in enumerate(zip(cur_para, ner_para)):
            entities = sent_ner
            entities_each_sent.append([entities, global_sent_id])

            if (cur_title, sent_id) in sp_set:
                sp_sents.append(global_sent_id)
                if answer_text in sent:
                    answer_labels.append(global_sent_id)
            para_sents.append(global_sent_id)
            meta_sents.append(sent)

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

    start_points = get_start_points_hotpot(entities_in_question, entities_each_sent, max_overlap_id)
    sent_edges = construct_graph_edge_by_entities(entity2sent)

    dict_wrap = {
        "sent_edges": sent_edges,
        "answer_labels": answer_labels,
        "start_points": list(start_points),
        "para_sents": paras,
        "sp_sents": sp_sents,
        "meta_sents": meta_sents,
        "meta_question": question_text,
        "_id": article_id
    }
    return dict_wrap


def preprocess_wikihop(article):
    article_id = article['id']
    paragraphs = article['supports']
    ner_paragraphs = article['ners']
    question_text = article['query'].strip().replace("\n", "")
    entities_in_question = article['question_entities']
    answer_text = article['answer'].strip().replace("\n", "")
    entities_each_sent = []
    sp_sents = []
    meta_sents = []
    answer_labels = []
    paras = []
    entity2sent = defaultdict(list)
    global_sent_id = 0
    max_overlap_id = 0
    max_rouge = 0
    start_points = []
    # print('question_text:', question_text)
    # print('answer_text:', answer_text)
    for para, para_ner in zip(paragraphs, ner_paragraphs):
        para_sents = []
        # cur_title, cur_para = para[0], para[1]
        # ner_title, ner_para = para_ner[0], para_ner[1]
        for sent_id, (sent, sent_ner) in enumerate(zip(para, para_ner)):
            entities = sent_ner
            entities_each_sent.append([entities, global_sent_id])
            if answer_text in sent.lower():
                answer_labels.append(global_sent_id)
            para_sents.append(global_sent_id)
            meta_sents.append(sent)

            if len(sent) > len(question_text) / 2:
                rouge_score = rouge.get_scores(sent, question_text)
                rf1 = rouge_score[0]['rouge-1']['f']
                if rf1 > max_rouge:
                    max_rouge = rf1
                    max_overlap_id = global_sent_id

            for e in entities:
                if global_sent_id not in entity2sent[e]:
                    entity2sent[e].append(global_sent_id)

            if len(entities_in_question[0]) > 0 and len(sent) > 5:
                # print(sent, entities_in_question)
                rouge_score = rouge.get_scores(sent.lower(), entities_in_question[0])
                rf1 = rouge_score[0]['rouge-1']['r']
                if rf1 > 0.5:
                    # print(sent, global_sent_id)
                    # print(rf1)
                    start_points.append(global_sent_id)

            global_sent_id += 1

        paras.append(para_sents)
    start_points.append(max_overlap_id)
    # start_points = get_start_points_hotpot(entities_in_question, entities_each_sent, max_overlap_id)
    sent_edges = construct_graph_edge_by_entities(entity2sent)

    dict_wrap = {
        "sent_edges": sent_edges,
        "answer_labels": answer_labels,
        "start_points": list(start_points),
        "para_sents": paras,
        "sp_sents": sp_sents,
        "meta_sents": meta_sents,
        "meta_question": question_text,
        "_id": article_id
    }
    return dict_wrap


def main(args):
    data = json.load(open(args.data_path))
    for article in tqdm(data):
        if args.task == 'hotpot':
            data_processed = preprocess_hotpot(article)
        elif args.task == 'wikihop':
            data_processed = preprocess_wikihop(article)
        else:
            data_processed = None
        sent_pairs = data_processed['sent_edges']
        entity_graph = build_graph(sent_pairs)
        start_points = data_processed['start_points']
        para_sents = data_processed['para_sents']
        answer_labels = data_processed['answer_labels']
        visited = [False] * 1000
        max_step = 6
        initial_step = 0
        initial_chain = []
        possible_chains = []

        # print("start_points:", start_points)
        # print('entity_graph:', entity_graph)
        # print('para_sents:', para_sents)
        # print('answer_labels:', answer_labels)
        search_all_possible_chain(start_points, entity_graph, para_sents, initial_chain, possible_chains,
                                  visited, initial_step, max_step, answer_labels)
        # print(possible_chains)
        if args.get_chain == "shortest":
            most_possible_chain = get_most_possible_chain_by_shortest_path(possible_chains, start_points)
        else:
            most_possible_chain = get_most_possible_chain_by_ques_overlap(
                possible_chains, start_points, data_processed['meta_sents'], data_processed['meta_question'])

        # print('most_possible_chain:', most_possible_chain)
        article['possible_chain'] = most_possible_chain
        # article['golden_head'] = []
        # article['coref_clusters'] = []
        # article['question'] = article.pop('query')
        # article['passage'] = article.pop('supports')
        # article['_id'] = article.pop('id')
        # print(possible_chains)
        # print("shortest:")
        # for i in most_possible_chain_shortest:
        #     print(i, ":", data_processed['meta_sents'][i])
        # print('sp_set:', data_processed['sp_sents'])
        # print()
        # print('--' * 30)
        #
        # print("overlap most:")
        # for i in most_possible_chain:
        #     print(i, ":", data_processed['meta_sents'][i])
        # print('sp_set:', data_processed['sp_sents'])
        # print()
        # print('--' * 30)

        # input()

    write_result(args.output_path, data)


def write_result(output_path, data):
    json.dump(data, open(output_path, 'w'))


def get_most_possible_chain_by_ques_overlap(possible_chains, start_points, meta_sentences, meta_question):
    max_rouge = 0
    best_chain_id = None
    for chain_id, chain in enumerate(possible_chains):
        sentences_in_chain = []
        for i in chain:
            sentences_in_chain.append(meta_sentences[i])
        concat_sents = " ".join(sentences_in_chain)
        rouge_score = rouge.get_scores(concat_sents, meta_question)
        rf1 = rouge_score[0]['rouge-l']['f'] + rouge_score[0]['rouge-1']['f'] + rouge_score[0]['rouge-2']['f']
        if rf1 > max_rouge:
            max_rouge = rf1
            best_chain_id = chain_id
    if best_chain_id is None:
        return []
    else:
        return possible_chains[best_chain_id]


def get_most_possible_chain_by_shortest_path(possible_chains, start_points, sp_set=None):
    most_possible_chain = []
    entity_entry = start_points[:-1]
    if len(possible_chains) > 0:
        min_len = len(possible_chains[0])
        most_possible_chain = possible_chains[0]
        for chain in possible_chains:
            # print(chain, entity_entry)
            # input()
            if len(chain) < min_len and chain[0] in entity_entry:
                min_len = len(chain)
                most_possible_chain = chain
    return most_possible_chain


def search_all_possible_chain(available_points, entity_graph, para_sents, current_chain, possible_chains,
                              visited, step, max_step, answer_labels):

    if step == max_step:
        return
    if len(set(current_chain).intersection(set(answer_labels))) > 0:
        possible_chains.append(list(current_chain))
        # print("find chain")
        # print("chain sets:", possible_chains)
        # print(available_points)
        # print(current_chain)
        # print(answer_labels)
        # print('---' * 20)
    else:
        # print("chain sets:", possible_chains)
        # print("available_points:", available_points)
        # print("current chain:", current_chain)
        # print(answer_labels)
        # print('---'*20)
        # input()
        for i in available_points:

            current_chain.append(i)
            visited[i] = True
            avl_points = find_all_available_points(i, entity_graph, para_sents, visited)
            search_all_possible_chain(avl_points, entity_graph, para_sents, current_chain, possible_chains,
                                      visited, step+1, max_step, answer_labels)
            current_chain.pop()
            visited[i] = False


def find_all_available_points(node, entity_graph, para_sents, visited):
    max_sent_in_para = 10
    available = []
    linked_nodes = entity_graph[node]
    # print('linked nodes:', linked_nodes)
    for n in linked_nodes:
        if not visited[n]:
            available.append(n)
    # print('node:', node)
    for para in para_sents:
        if node in para:
            for n in para[:max_sent_in_para]:
                if not visited[n]:
                    available.append(n)
    # print(available)
    return available


def check_answer(chain, answer_set):
    print("intersection:", answer_set.intersection(chain))
    return 1 if len(answer_set.intersection(chain)) > 0 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('--data_path', type=str, help='path to the input data file')
    parser.add_argument('--output_path', type=str, help='path to the output data file', default="")
    parser.add_argument('--get_chain', type=str, help='criteria to get the chain', default="shortest")
    parser.add_argument('--task', type=str, help='which task wikihop/hotpot', default='hotpot')
    args = parser.parse_args()
    main(args)
