import json
import argparse
from collections import defaultdict


def reduce_by_top_k_sent_wikihop(args_parse, original_data, top_k=5):
    para_sent_count = defaultdict(int)
    passage_para_count = defaultdict(int)
    answer_find = 0
    total_instance = 0

    for original_instance in original_data:
        selected_paras = []
        selected_ners = []
        total_instance += 1
        ans = original_instance['answer']
        original_paragraphs = original_instance['passage']
        passage_para_count[len(original_paragraphs)] += 1
        original_ners = original_instance['ners']
        for i, para in enumerate(original_paragraphs[:20]):
            para_sent_count[len(para)] += 1
            para = para[:top_k]
            selected_paras.append(para)
            # if ans.lower() in " ".join(para).lower():
            #     answer_find += 1
            #     break
        for para_ners in original_ners:
            para_ners = para_ners[:top_k]
            selected_ners.append(para_ners)
        original_instance['query'] = original_instance.pop('question')
        original_instance.pop('passage')
        original_instance['id'] = original_instance.pop('_id')
        original_instance['supports'] = selected_paras
        original_instance['ners'] = selected_ners
    for i in sorted(para_sent_count):
        print(i, ":", para_sent_count[i])
    print('*' * 100)
    for i in sorted(passage_para_count):
        print(i, ":", passage_para_count[i])
    print("Total instances : {}".format(total_instance))
    print("Answer find in {} instances".format(answer_find))
    json.dump(original_data, open(args_parse.output_path, 'w'), indent=4)


def reduce_by_top_5_sent_hotpot(args_parse, original_data):

    for original_instance in original_data:
        original_paragraphs = original_instance['context']
        original_ners = original_instance['ners']
        for i, para in enumerate(original_paragraphs):
            para[1] = para[1][:5]
        for para_ners in original_ners:
            para_ners = para_ners[:5]

        original_instance['golden_head'] = []
        original_instance['coref_clusters'] = []

    json.dump(original_data, open(args_parse.output_path, 'w'), indent=4)


def reduce_by_selected_paras_hotpot(args_parse, original_data, selected_data):
    # selected_instance_ids = selected_data.keys()
    para_num = 0
    sent_num = 0
    max_sent_num = 0
    sp_sent_num = 0
    sp_sent_id_sum = 0
    for original_instance in original_data:
        original_instance_id = original_instance['_id']
        original_paragraphs = original_instance['context']
        original_ners = original_instance['ners']
        original_sp_set = original_instance['supporting_facts']
        correspond_selected_instance = selected_data[original_instance_id]
        selected_titles = [para[0] for para in correspond_selected_instance]
        sp_titles = [sp[0] for sp in original_sp_set]
        sp_sent_id = [sp[1] for sp in original_sp_set]
        sp_sent_id_sum += len([idx for idx in sp_sent_id if idx > 5])
        sp_sent_num += len(sp_sent_id)
        selected_idx = []
        for i, para in enumerate(original_paragraphs):
            # para_num += 1
            # sent_num += len(para[1])
            # if len(para[1]) > max_sent_num:
            #     max_sent_num = len(para[1])
            cur_title, cur_para = para[0], para[1]
            if cur_title in selected_titles:
                selected_idx.append(i)
            if cur_title in sp_titles:
                para_num += 1
                sent_num += len(para[1])
                if len(para[1]) > 6:
                    max_sent_num += 1

        def truncate_too_long_para(paras):
            for para in paras:
                para[1] = para[1][:5]

        truncate_too_long_para(original_paragraphs)
        truncate_too_long_para(original_ners)
        selected_paras = [original_paragraphs[i] for i in selected_idx]
        selected_ners = [original_ners[i] for i in selected_idx]
        if any([len(para) for para in selected_paras]) > 5:
            print("ERROR")
            input()
        original_instance['context'] = selected_paras
        original_instance['ners'] = selected_ners
        original_instance['golden_head'] = []
        original_instance['coref_clusters'] = []
    print(para_num)
    print(sent_num)
    print(max_sent_num)
    print(float(sent_num) / para_num)
    print(sp_sent_id_sum)
    print(sp_sent_num)
    print(sp_sent_id_sum / float(sp_sent_num))
    json.dump(original_data, open(args_parse.output_path, 'w'), indent=4)


def main(args_parse):
    original_data = json.load(open(args_parse.original_path, 'r'))
    selected_data = None
    if args_parse.selected_path is not None:
        selected_data = json.load(open(args_parse.selected_path, 'r'))
    # reduce_by_selected_paras_hotpot(args_parse, original_data, selected_data)
    reduce_by_top_k_sent_wikihop(args_parse, original_data)
    # reduce_by_top_5_sent(args_parse, original_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label golden dependency heads map for hotpot dataset')
    parser.add_argument('--original_path', help='path to hotpot dataset')
    parser.add_argument('--selected_path', default=None, help='path to the selected paragraph')
    parser.add_argument('--output_path', help='path to dep-labeled hotpot dataset')
    args = parser.parse_args()
    main(args)
