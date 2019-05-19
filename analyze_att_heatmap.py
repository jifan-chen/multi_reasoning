import my_library
import argparse
import torch
from allennlp.models.archival import load_archive
from my_library.predictors import HotpotPredictor
from my_library.metrics import AttF1Measure
import numpy as np
from allennlp.tools import squad_eval
import json, pickle, os, sys
from tqdm import tqdm
import random
import heapq
import shutil
from distutils.util import strtobool


TH = 0.2


def make_save_dir(save_dir_path):
    if os.path.exists(save_dir_path):
        print('----save dir {} already exist.'.format(save_dir_path))
        while True:
            try:
                ans = strtobool(input('----Do you actually want to remove {}? (Y/n)'.format(save_dir_path)))
                if ans:
                    shutil.rmtree(save_dir_path, ignore_errors=True)
                    break
            except ValueError:
                print('----Please respond with \'y\' or \'n\'.\n')
    assert not os.path.exists(save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True)


def find_att_toks(scores, mask, th, row_offset, conn_type):
    scores = scores * mask
    accept = scores >= th
    accept_row = np.sum(accept, axis=1) >= 1
    accept_scores = scores * accept
    row_idx = (np.arange(scores.shape[0])[accept_row] + row_offset).tolist()
    row_accept_scores = accept_scores[accept_row, :].tolist()
    return list(map(lambda x: {'target': "", 'pos': x[0], 'scores': x[1], 'type': conn_type},
                    zip(row_idx, row_accept_scores)))


def calc_em_and_f1(best_span_string, answer_strings):
    exact_match = squad_eval.metric_max_over_ground_truths(
            squad_eval.exact_match_score,
            best_span_string,
            answer_strings)
    f1_score = squad_eval.metric_max_over_ground_truths(
            squad_eval.f1_score,
            best_span_string,
            answer_strings)
    return exact_match, f1_score


def PRF1(T_P, N_P, N_T):
    precision = float(T_P) / float(N_P + 1e-13)
    recall = float(T_P) / float(N_T + 1e-13)
    f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure


def map_score_to_metric(scores, labels, metric, metric_table):
    # metric should be from 0 ~ 1
    table_idx = int(metric * 100) if metric < 1. else 99
    min_scores = np.min(scores, axis=1)
    max_scores = np.max(scores, axis=1)
    sum_scores = np.sum(scores, axis=1)
    mean_scores = np.mean(scores, axis=1)
    std_scores = np.std(scores, axis=1)
    ent_scores = -np.sum(np.log(scores / sum_scores[:, None] + 1e-10), axis=1)
    T_conn_scores = scores[labels]
    F_conn_scores = scores[~labels]
    metric_table[table_idx]['min'].extend(min_scores)
    metric_table[table_idx]['max'].extend(max_scores)
    metric_table[table_idx]['sum'].extend(sum_scores)
    metric_table[table_idx]['mean'].extend(mean_scores)
    metric_table[table_idx]['std'].extend(std_scores)
    metric_table[table_idx]['ent'].extend(ent_scores)
    metric_table[table_idx]['T_connection'].extend(T_conn_scores)
    metric_table[table_idx]['F_connection'].extend(F_conn_scores)
    metric_table[table_idx]['num_maps'] += 1


def analyze_att(att_scores, labels, metric, metric_tables, row_spans, col_spans, num_att_heads):
    self_mask = 1 - np.identity(att_scores.shape[1])
    all_att_toks = [{'T': [], 'F': []} for h_idx in range(num_att_heads)]
    for (r_s, r_e), (c_s, c_e) in zip(row_spans, col_spans):
        att_scores_sp = att_scores[:, r_s:r_e+1, c_s:c_e+1]
        labels_sp = labels[r_s:r_e+1, c_s:c_e+1]
        self_mask_sp = self_mask[r_s:r_e+1, c_s:c_e+1]
        for h_idx in range(num_att_heads):
            map_score_to_metric(att_scores_sp[h_idx], labels_sp, metric, metric_tables[h_idx])
            att_toks = find_att_toks(att_scores_sp[h_idx], labels_sp*self_mask_sp, TH, r_s, 'T')
            all_att_toks[h_idx]['T'].extend(att_toks)
            att_toks = find_att_toks(att_scores_sp[h_idx], (~labels_sp)*self_mask_sp, TH, r_s, 'F')
            all_att_toks[h_idx]['F'].extend(att_toks)
    table_idx = int(metric * 100) if metric < 1. else 99
    for h_idx in range(num_att_heads):
        metric_tables[h_idx][table_idx]['num_instances'] += 1
    return all_att_toks


def split_metric_table_by_quartile(metric_table):
    num_instances = sum([m['num_instances'] for m in metric_table])
    num_quartile = np.ceil(num_instances / 4)
    clusters = [{'stat': new_stat_dict(),
                 'start': None,
                 'end': None}]
    q_idx = 1
    num_seen_instances = 0
    for m_idx, m in enumerate(metric_table):
        if m['num_instances'] > 0:
            if num_seen_instances >= q_idx*num_quartile:
                clusters.append({'stat': new_stat_dict(),
                                 'start': None,
                                 'end': None})
                q_idx += 1
            clusters[-1]['stat']['min'].extend(m['min'])
            clusters[-1]['stat']['max'].extend(m['max'])
            clusters[-1]['stat']['sum'].extend(m['sum'])
            clusters[-1]['stat']['mean'].extend(m['mean'])
            clusters[-1]['stat']['std'].extend(m['std'])
            clusters[-1]['stat']['ent'].extend(m['ent'])
            clusters[-1]['stat']['T_connection'].extend(m['T_connection'])
            clusters[-1]['stat']['F_connection'].extend(m['F_connection'])
            clusters[-1]['stat']['num_maps'] += m['num_maps']
            clusters[-1]['stat']['num_instances'] += m['num_instances']
            if clusters[-1]['start'] is None:
                clusters[-1]['start'] = m_idx
                clusters[-1]['end'] = m_idx
            else:
                clusters[-1]['end'] = m_idx
            num_seen_instances += m['num_instances']
    return clusters


def get_dep_map(dep_connections, seq_len):
    m = np.zeros((seq_len, seq_len))
    m[[i for i, j in dep_connections], [j for i, j in dep_connections]] = 1
    return m


def get_coref_map(coref_clusters, seq_len, passage_tokens):
    m = np.zeros((seq_len, seq_len))
    for c in coref_clusters:
        for i in range(0, len(c)-1):
            for j in range(i+1, len(c)):
                i_s, i_e = c[i]
                j_s, j_e = c[j]
                '''
                if not " ".join(passage_tokens[i_s:i_e+1]).lower() == " ".join(passage_tokens[j_s:j_e+1]).lower():
                    m[i_s:i_e+1, j_s:j_e+1] = 1
                    m[j_s:j_e+1, i_s:i_e+1] = 1
                '''
                m[i_s:i_e+1, j_s:j_e+1] = 1
                m[j_s:j_e+1, i_s:i_e+1] = 1
    return m


def new_stat_dict():
    return {'min': [],
            'max': [],
            'sum': [],
            'mean': [],
            'std': [],
            'ent': [],
            'T_connection': [],
            'F_connection': [],
            'num_maps': 0,
            'num_instances': 0}


def new_metric_table():
    metric_table = [new_stat_dict() for _ in range(100)]
    return metric_table


def print_stat(stat, metric_start, metric_end, metric_name, file=sys.stdout):
    metric_start = metric_start / 100
    metric_end = (metric_end + 1) / 100
    T_conn_bins = np.linspace(min(stat['T_connection']), max(stat['T_connection']), 11)
    T_hist, _ = np.histogram(stat['T_connection'], T_conn_bins)
    T_density = [h / len(stat['T_connection'])*100 for h in T_hist]
    F_conn_bins = np.linspace(min(stat['F_connection']), max(stat['F_connection']), 11)
    F_hist, _ = np.histogram(stat['F_connection'], F_conn_bins)
    F_density = [h / len(stat['F_connection'])*100 for h in F_hist]
    print("===================================", file=file)
    print("{}: {:.2f} ~ {:.2f}".format(metric_name, metric_start, metric_end), file=file)
    print("N Connections:   {}".format(len(stat['T_connection'])+len(stat['F_connection'])), file=file)
    print("N Tokens:        {}".format(len(stat['min'])), file=file)
    print("N Supports:      {}".format(stat['num_maps']), file=file)
    print("N Instances:     {}".format(stat['num_instances']), file=file)
    print("-----------------------------------", file=file)
    print("Min:     avg {:.2e} - std {:.2e}".format(np.mean(stat['min']), np.std(stat['min'])), file=file)
    print("Max:     avg {:.2e} - std {:.2e}".format(np.mean(stat['max']), np.std(stat['max'])), file=file)
    print("Sum:     avg {:.2e} - std {:.2e}".format(np.mean(stat['sum']), np.std(stat['sum'])), file=file)
    print("Mean:    avg {:.2e} - std {:.2e}".format(np.mean(stat['mean']), np.std(stat['mean'])), file=file)
    print("Std:     avg {:.2e} - std {:.2e}".format(np.mean(stat['std']), np.std(stat['std'])), file=file)
    print("Entropy: avg {:.2e} - std {:.2e}".format(np.mean(stat['ent']), np.std(stat['ent'])), file=file)
    print("-----------------------------------", file=file)
    print("True Connection Histogram", file=file)
    print("|"+"|".join(" {:.2e}~ ".format(T_conn_bins[i]) for i in range(10))+"|", file=file)
    print("|"+"|".join(" {:.2e}  ".format(T_conn_bins[i+1]) for i in range(10))+"|", file=file)
    print("|"+"|".join(" {:9d} ".format(T_hist[i]) for i in range(10))+"|", file=file)
    print("|"+"|".join(" {:8.2f}% ".format(T_density[i]) for i in range(10))+"|", file=file)
    print("| avg {:.2e} - std {:.2e}".format(np.mean(stat['T_connection']), np.std(stat['T_connection'])), file=file)
    print("-----------------------------------", file=file)
    print("False Connection Histogram", file=file)
    print("|"+"|".join(" {:.2e}~ ".format(F_conn_bins[i]) for i in range(10))+"|", file=file)
    print("|"+"|".join(" {:.2e}  ".format(F_conn_bins[i+1]) for i in range(10))+"|", file=file)
    print("|"+"|".join(" {:9d} ".format(F_hist[i]) for i in range(10))+"|", file=file)
    print("|"+"|".join(" {:8.2f}% ".format(F_density[i]) for i in range(10))+"|", file=file)
    print("| avg {:.2e} - std {:.2e}".format(np.mean(stat['F_connection']), np.std(stat['F_connection'])), file=file)
    print("===================================", file=file)


def gather_coref_clusters(coref_instance, para_limit, bypara):
    if bypara:
        coref_doc = []
        coref_clusters = []
        for title, para_coref in coref_instance['coref_info']:
            offset = len(coref_doc)
            for c in para_coref['clusters']:
                shifted_c = [[s+offset, e+offset] for s, e in c if s+offset < para_limit and e+offset < para_limit]
                if len(shifted_c) > 1:
                    coref_clusters.append(shifted_c)
            coref_doc.extend(para_coref['document'])
    else:
        coref_doc = coref_instance['coref_info']['document']
        coref_clusters = []
        for c in coref_instance['coref_info']['clusters']:
            filtered_c = [[s, e] for s, e in c if s < para_limit and e < para_limit]
            if len(filtered_c) > 1:
                coref_clusters.append(filtered_c)
    poses = []
    tags = []
    for title, para_pos, para_tag in coref_instance['pos']:
        for sent_pos in para_pos:
            poses.extend(sent_pos)
        for sent_tag in para_tag:
            tags.extend(sent_tag)
    return coref_doc, coref_clusters, poses, tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze self attention heap maps')
    parser.add_argument('model_path', type=str, help='path to the tgz model file')
    parser.add_argument('data_path', type=str, help='path to the data file')
    parser.add_argument('att_dir', type=str, help='path to dir to store the att tokens dicts')
    args = parser.parse_args()

    make_save_dir(args.att_dir)
    logfile = open(os.path.join(args.att_dir, 'res.txt'), 'w')

    archive = load_archive(args.model_path, cuda_device=0)
    predictor = HotpotPredictor.from_archive(archive, 'hotpot_predictor')
    para_limit = predictor._dataset_reader._para_limit

    num = 0
    tot_em = 0.
    tot_f1 = 0.
    num_att_heads = 2
    att_f1_metrics = [[AttF1Measure(TH) for _ in range(4)] for h_idx in range(num_att_heads)]
    coref_f1_tables = [new_metric_table() for _ in range(num_att_heads)]
    all_att_toks = [{'T': [], 'F': []} for _ in range(num_att_heads)]
    all_passage_tokens = []
    all_question_tokens = []
    all_answer_texts = []
    all_pred_ans = []
    all_f1 = []
    w_cnt = 0
    g_sc = 0
    ng_sc = 0
    print(" ".join(["{:10s}"]*11).format("Idx", "Head 0", "F1", "P Num", "T Num", "T-P",
                                                "Head 1", "F1", "P Num", "T Num", "T-P"), file=logfile)
    for instance in tqdm(predictor._dataset_reader.read(args.data_path)):
        num += 1
        # prediction res
        output = predictor.predict_instance(instance)
        loss                = output['loss']
        passage_tokens      = output['passage_tokens']
        question_tokens     = output['question_tokens']
        token_spans_sp      = output['token_spans_sp']
        token_spans_sent    = output['token_spans_sent']
        sent_labels         = output['sent_labels']
        coref_clusters      = output['coref_clusters']
        self_att_scores     = output['self_attention_score']
        answer_texts        = output['answer_texts']
        best_span_str       = output['best_span_str']
        article_id          = output['_id']
        sp_mask = np.zeros(len(passage_tokens), dtype='int')
        for s, e in token_spans_sp:
            sp_mask[s:e] = 1
        sp_sp_mask = sp_mask[None, :] & sp_mask[:, None]
        sp_all_mask = sp_mask[None, :] | sp_mask[:, None]
        em, f1 = calc_em_and_f1(best_span_str, answer_texts)
        tot_em += em
        tot_f1 += f1
        # coref res
        coref_map = get_coref_map(coref_clusters, len(passage_tokens), passage_tokens)
        assert np.array(self_att_scores).shape == (2, len(passage_tokens), len(passage_tokens))
        #assert np.allclose(np.sum(np.array(self_att_scores), axis=2), 1.)
        assert len(sent_labels) == len(token_spans_sent)
        for i, (s, e) in enumerate(token_spans_sent):
            if not i == 0:
                assert s == token_spans_sent[i-1][1] + 1
            if not i == len(token_spans_sent) - 1:
                assert e == token_spans_sent[i+1][0] - 1
        '''
        print('support:')
        for s, e in token_spans_sp:
            print(" ".join(passage_tokens[s:e+1]))
        print('label support:')
        for i, l in enumerate(sent_labels):
            if l == 1:
                s, e = token_spans_sent[i]
                print(" ".join(passage_tokens[s:e+1]))
        print('sentence:')
        for i, (s, e) in enumerate(token_spans_sent):
            print(" ".join(passage_tokens[s:e+1]))
        print('sent_labels:', sent_labels)
        print('attn shape:', np.array(self_att_scores).shape, len(passage_tokens))
        print('answer_texts:', answer_texts)
        print('best_span_str:', best_span_str)
        print('article_id:', article_id)
        print('coref:')
        for c in coref_clusters:
            for s, e in c:
                print("'{} - {}'".format(" ".join(passage_tokens[s:e+1]), " ".join(tags[s:e+1])), end=' ')
            print()
        if num >= 0:
            break
        '''
        # coref analysis
        self_att_scores = np.array(self_att_scores)
        coref_map = coref_map.astype(bool)
        col_spans = [(0, len(passage_tokens)-1)]*len(token_spans_sp)
        cur_att_toks = analyze_att(self_att_scores, coref_map, f1, coref_f1_tables, token_spans_sp, col_spans, num_att_heads)
        # find att tokens
        for h_idx in range(num_att_heads):
            all_att_toks[h_idx]['T'].append([num-1, cur_att_toks[h_idx]['T']])
            all_att_toks[h_idx]['F'].append([num-1, cur_att_toks[h_idx]['F']])
        all_passage_tokens.append(passage_tokens)
        all_question_tokens.append(question_tokens)
        all_answer_texts.append(answer_texts)
        all_f1.append(f1)
        all_pred_ans.append(best_span_str)
        # att fit
        val_w = np.sum(coref_map, axis=-1) > 0
        g_sc_per = np.sum((self_att_scores * coref_map), axis=-1) * val_w
        ng_sc_per = np.sum((self_att_scores * (1-coref_map)), axis=-1) * val_w
        assert g_sc_per.shape[1] == ng_sc_per.shape[1]
        w_cnt += np.sum(val_w)
        g_sc += np.sum(g_sc_per, axis=-1)
        ng_sc += np.sum(ng_sc_per, axis=-1)
        # att f1 analysis
        self_att_scores = torch.tensor(self_att_scores).float()
        coref_map = torch.tensor(coref_map.astype('float')).float()
        sp_sp_mask = torch.tensor(sp_sp_mask).float()
        sp_all_mask = torch.tensor(sp_all_mask).float()
        line = "{:<10d} ".format(num-1)
        for h_idx in range(num_att_heads):
            line += "{:10s} ".format(" ")
            ssT_P, ssN_P, ssN_T = att_f1_metrics[h_idx][0](self_att_scores[h_idx]*sp_sp_mask, coref_map*sp_sp_mask)
            saT_P, saN_P, saN_T = att_f1_metrics[h_idx][1](self_att_scores[h_idx]*sp_all_mask, coref_map*sp_all_mask)
            nssT_P, nssN_P, nssN_T = att_f1_metrics[h_idx][2](self_att_scores[h_idx]*(1-sp_sp_mask), coref_map*(1-sp_sp_mask))
            nsaT_P, nsaN_P, nsaN_T = att_f1_metrics[h_idx][3](self_att_scores[h_idx]*(1-sp_all_mask), coref_map*(1-sp_all_mask))
            T_P = ssT_P + saT_P + nssT_P + nsaT_P
            N_P = ssN_P + saN_P + nssN_P + nsaN_P
            N_T = ssN_T + saN_T + nssN_T + nsaN_T
            _, _, att_f1 = PRF1(T_P, N_P, N_T)
            line += "{:<10.4f} {:<10.1f} {:<10.1f} {:<10.1f} ".format(att_f1, N_P, N_T, T_P)
        print(line, file=logfile)
    # display attention scores statistics
    coref_f1_res = [split_metric_table_by_quartile(tab) for tab in coref_f1_tables]
    print('\n\ndata Num:', num, 'Avg EM:', tot_em / num, 'Avg F1:', tot_f1 / num, file=logfile)
    print("Coref", file=logfile)
    for i, res in enumerate(coref_f1_res):
        print("Head %d" % i, file=logfile)
        for c in res:
            print_stat(c['stat'], c['start'], c['end'], 'F1', file=logfile)
        print(file=logfile)
    print(file=logfile)

    type_name = ["SP-SP", "SP-ALL", "~SP-SP", "~SP-ALL"]
    print(" ".join(["{:10s}"]*11).format("Type", "Head 0", "F1", "P Num", "T Num", "T-P",
                                                 "Head 1", "F1", "P Num", "T Num", "T-P"), file=logfile)
    for type_idx in range(4):
        line = "{:10s} ".format(type_name[type_idx])
        for h_idx in range(num_att_heads):
            line += "{:10s} ".format("Head %d" % h_idx)
            T_P = att_f1_metrics[h_idx][type_idx]._true_positives.item()
            F_P = att_f1_metrics[h_idx][type_idx]._false_positives.item()
            F_N = att_f1_metrics[h_idx][type_idx]._false_negatives.item()
            _, _, att_f1 = PRF1(T_P, T_P+F_P, T_P+F_N)
            line += "{:<10.4f} {:<10.1f} {:<10.1f} {:<10.1f} ".format(att_f1, T_P+F_P, T_P+F_N, T_P)
        print(line+'\n', file=logfile)

    print("att fit", file=logfile)
    print(g_sc / w_cnt, file=logfile)
    print(ng_sc / w_cnt, file=logfile)
    logfile.close()

    '''
    # sample and store att tokens
    def KMostExamples(iterable, seen_doc_idxs):
        att_res = []
        for pair in iterable:
            assert pair[0][0] == pair[1][0]
            if pair[0][0] in seen_doc_idxs:
                continue
            if len(pair[0][1]) == 0 and len(pair[1][1]) == 0:
                continue
            seen_doc_idxs.add(pair[0][0])
            doc = all_passage_tokens[pair[0][0]]
            q = all_question_tokens[pair[0][0]]
            ans = all_answer_texts[pair[0][0]]
            pred = all_pred_ans[pair[0][0]]
            f1 = all_f1[pair[0][0]]
            doc_res = []
            for tok_dict in pair[0][1]:
                assert len(tok_dict['target']) == 0
                tok_dict['target'] = doc[tok_dict['pos']]
                doc_res.append(tok_dict)
            for tok_dict in pair[1][1]:
                assert len(tok_dict['target']) == 0
                tok_dict['target'] = doc[tok_dict['pos']]
                doc_res.append(tok_dict)
            att_res.append({'doc': doc,
                            'attns': doc_res,
                            'question': " ".join(q),
                            'answer': " ".join(ans),
                            'predict': pred,
                            'f1': f1})
        return att_res
    for h_idx in range(num_att_heads):
        print("Total att toks for T for head %d:" % h_idx, sum([len(ele[1]) for ele in all_att_toks[h_idx]['T']]))
        print("Total att toks for F for head %d:" % h_idx, sum([len(ele[1]) for ele in all_att_toks[h_idx]['F']]))
        att_res = []
        seen_doc_idxs = set()
        # doc with most number of T examples
        iterable = heapq.nlargest(3,
                                  zip(all_att_toks[h_idx]['T'], all_att_toks[h_idx]['F']),
                                  key=lambda x: len(x[0][1]))
        att_res += KMostExamples(iterable, seen_doc_idxs)
        # doc with most number of F examples
        iterable = heapq.nlargest(3,
                                  zip(all_att_toks[h_idx]['T'], all_att_toks[h_idx]['F']),
                                  key=lambda x: len(x[1][1]))
        att_res += KMostExamples(iterable, seen_doc_idxs)
        # doc with most number of T+F examples
        iterable = heapq.nlargest(3,
                                  zip(all_att_toks[h_idx]['T'], all_att_toks[h_idx]['F']),
                                  key=lambda x: len(x[0][1])+len(x[1][1]))
        att_res += KMostExamples(iterable, seen_doc_idxs)
        print("Total instances stored for head %d:" % h_idx, len(att_res))
        with open(os.path.join(args.att_dir, 'heads%d.json' % h_idx), 'w') as f:
            json.dump(att_res, f, indent=4)
    '''
