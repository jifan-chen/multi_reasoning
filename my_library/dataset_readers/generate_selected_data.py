import json
import argparse


def reduce_by_top_5_sent(args_parse, original_data):

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


def reduce_by_selected_paras(args_parse, original_data, selected_data):
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
        selected_paras = [original_paragraphs[i][:5] for i in selected_idx]
        selected_ners = [original_ners[i][:5] for i in selected_idx]
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
    selected_data = json.load(open(args_parse.selected_path, 'r'))
    reduce_by_selected_paras(args_parse, original_data, selected_data)
    # reduce_by_top_5_sent(args_parse, original_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label golden dependency heads map for hotpot dataset')
    parser.add_argument('--original_path', help='path to hotpot dataset')
    parser.add_argument('--selected_path', help='path to the selected paragraph')
    parser.add_argument('--output_path', help='path to dep-labeled hotpot dataset')
    args = parser.parse_args()
    main(args)
