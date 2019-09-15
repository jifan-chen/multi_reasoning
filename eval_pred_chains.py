import json
from tqdm import tqdm
import argparse
import glob
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from allennlp.training.metrics import SquadEmAndF1
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
#from  nltk.translate.bleu_score import sentence_bleu


PUNC = string.punctuation
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=False)

def process_sent_tokens(sent_tokens):
    sent_tokens = tokenizer.split_words(" ".join(sent_tokens))
    return [tok.lemma_.lower() for tok in sent_tokens if not tok.text.lower() in STOPLIST and not tok.text.lower() in PUNC]
    #return [tok.lemma_.lower() for tok in sent_tokens]


def num_TP_sp(chain, sp_list, q_lemmas=None, p_lemmas=None):
    sp_set = set(sp_list)
    chain_set = set(chain)
    return len(chain_set & sp_set)


def num_TP_q(chain, sp_list, q_lemmas, p_lemmas):
    chain_toks = []
    for s_idx in chain:
        chain_toks += p_lemmas[s_idx]
    chain_toks = set(chain_toks)
    q_toks = set(q_lemmas)
    return len(chain_toks & q_toks)
    #return sentence_bleu([q_lemmas], chain_toks, weights=(0.5, 0.5, 0, 0))
    #p = len(chain_toks & q_toks) / (len(chain_toks) + 1e-13)
    #r = len(chain_toks & q_toks) / (len(q_toks) + 1e-13)
    #return 2 * p * r / (p + r + 1e-13)


def get_necessary_info(article):
    paragraphs = article['context']
    answer_text = article['answer'].strip().replace("\n", "")
    sp_set = set(list(map(tuple, article['supporting_facts'])))
    sent_labels = []
    ans_sent_idxs = []
    for para in paragraphs:
        cur_title, cur_para = para[0], para[1]
        for sent_id, sent in enumerate(cur_para):
            if (cur_title, sent_id) in sp_set:
                if answer_text in sent:
                    ans_sent_idxs.append(len(sent_labels))
                sent_labels.append(1)
            else:
                sent_labels.append(0)
    article['sent_labels'] = sent_labels
    article['ans_sent_idxs'] = ans_sent_idxs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate chain predictions')
    parser.add_argument('pred_path', type=str, help='path to prediction file')
    parser.add_argument('--data_path', type=str, help='path to orig data file')
    parser.add_argument('--k', type=int, help='the number of predicted chains to consider',
                        default=100000)
    parser.add_argument('--per_line', default=False, help='whether the prediction is stored per line in the prediction file',
                        action='store_true')
    parser.add_argument('--sort', default='sp', help='whether the sorting is by sp set or overlap wrt question',
                        choices=['sp', 'q'])
    parser.add_argument('--filter', default='', help='whether filter out the branching and yes-no question',
                        choices=['branch'])
    parser.add_argument('--eval_chain_type', default='pred', help='the type of chain to evaluate',
                        choices=['pred', 'oracle'])
    parser.add_argument('--chain_f1_target', default='sp', help='to use rb or sp when calc the chain f1',
                        choices=['sp', 'rb'])
    args = parser.parse_args()

    res_path = args.pred_path
    k = args.k

    # read prediction results
    res = []
    for fn in glob.glob(res_path):
        with open(fn, 'r') as f:
            if args.per_line:
                for line in f:
                    res.append(json.loads(line))
            else:
                res += json.load(f)
    # get sent_labels, etc if needed. Here we assume the prediction is data file
    if not 'sent_labels' in res[0]:
        for r in res:
            get_necessary_info(r)
    print("Number of instances:", len(res))
    print("Number of ids:", len(set([r['_id'] for r in res])))
    
    if args.eval_chain_type == 'oracle':
        for r in res:
            r['pred_chains'] = [r['possible_chain']]

    # filter out chain s_idx that is greater then the number of sent (dev 4962)
    if 'sent_labels' in res[0]:
        for r in res:
            for i, c in enumerate(r['pred_chains']):
                r['pred_chains'][i] = [s_idx for s_idx in c if s_idx < len(r['sent_labels'])]
    
    if args.filter == 'branch':
        # filter branching question
        res_filter = []
        for r in res:
            if not r['answer_texts'][0] in r['question_tokens'] and not r['answer_texts'][0] in ['yes', 'no']:
                res_filter.append(r)
        res = res_filter
        print("Number of instances:", len(res))
        print("Number of ids:", len(set([r['_id'] for r in res])))

    # get the sorted prediction results
    if 'sent_labels' in res[0]:
        if args.sort == 'q':
            num_TP = num_TP_q
            for r in tqdm(res):
                r['question_lemma'] = process_sent_tokens(r['question_tokens'])
                r['sent_lemmas'] = []
                for s, e in r['token_spans_sent']:
                    r['sent_lemmas'].append(process_sent_tokens(r['passage_tokens'][s:e+1]))
        else:
            for r in tqdm(res):
                r['question_lemma'] = None
                r['sent_lemmas'] = None
            num_TP = num_TP_sp
        for r in res:
            sp_list = [i for i, ind in enumerate(r['sent_labels']) if ind == 1]
            pred_chains = sorted(r['pred_chains'], key=lambda x: num_TP(x, sp_list, r['question_lemma'], r['sent_lemmas']), reverse=True)
            r['sorted_pred_chains'] = pred_chains

    if args.data_path:
        data = []
        for fn in glob.glob(args.data_path):
            with open(fn, 'r') as f:
                data += json.load(f)
        print("Number of data instances:", len(data))
        print("Number of data ids:", len(set([d['_id'] for d in data])))
        data_id2idx = {d['_id']: i for i, d in enumerate(data)}

    if args.data_path and 'best_span_str' in res[0]:
        squad_metrics = SquadEmAndF1()
        for r in res:
            best_span_str = r['best_span_str']
            answer = data[data_id2idx[r['_id']]]['answer'].strip().replace("\n", "")
            squad_metrics(best_span_str, [answer])
        em, f1 = squad_metrics.get_metric(reset=True)
        print("Ans EM:", em, "Ans F1:", f1)


    if 'ans_sent_idxs' in res[0] and not res[0]['ans_sent_idxs'] is None:
        # ans include in prediction
        num = 0
        corr = 0
        for r in res:
            if len(r['ans_sent_idxs']) > 0:
                num += 1
                if len(set([s for b in r['pred_chains'][:k] for s in b]) & set(r['ans_sent_idxs'])) > 0:
                    corr += 1
        print("ans include in top k prediction:", corr / num)

        if 'sent_labels' in res[0]:
            # ans include in sorted prediction
            num = 0
            corr = 0
            for r in res:
                if len(r['ans_sent_idxs']) > 0:
                    pred_chains = r['sorted_pred_chains']
                    num += 1
                    if len(set([s for b in pred_chains[:k] for s in b]) & set(r['ans_sent_idxs'])) > 0:
                        corr += 1
            print("ans include in sorted top k prediction:", corr / num)


    if 'possible_chain' in res[0] and res[0]['possible_chain']:
        # chain em w.r.t possible chain
        num = 0
        corr = 0
        for r in res:
            if len(r['possible_chain']) > 0:
                num += 1
                if any([b == r['possible_chain'] for b in r['pred_chains'][:k]]):
                    corr += 1
        print("top k chain em wrt gold chain:", corr / num)

        if 'sent_labels' in res[0]:
            # sorted chain em w.r.t possible chain
            num = 0
            corr = 0
            for r in res:
                if len(r['possible_chain']) > 0:
                    pred_chains = r['sorted_pred_chains']
                    num += 1
                    if any([b == r['possible_chain'] for b in pred_chains[:k]]):
                        corr += 1
            print("top k sorted chain em wrt gold chain:", corr / num)


    if 'sent_labels' in res[0]:
        # chain union f1 wrt sp set
        TP = 0
        NP = 0
        NT = 0
        for r in res:
            pred_chains_flat = [s for b in r['pred_chains'][:k] for s in b]
            if args.chain_f1_target == 'sp':
                sp_list = [i for i, ind in enumerate(r['sent_labels']) if ind == 1]
            else:
                sp_list = r['possible_chain']
            TP += num_TP_sp(pred_chains_flat, sp_list)
            NP += len(set(pred_chains_flat))
            NT += len(sp_list)
        prec = TP / NP;
        recl = TP / NT;
        f1 = 2 * prec * recl / (prec + recl)
        print("union prediction f1 wrt %s set:" % args.chain_f1_target)
        print("Prec:", prec, "Recl:", recl, "F1:", f1)


        # f1 between the union of topk best TP chain and sp set
        TP = 0
        NP = 0
        NT = 0
        for r in res:
            if args.chain_f1_target == 'sp':
                sp_list = [i for i, ind in enumerate(r['sent_labels']) if ind == 1]
            else:
                sp_list = r['possible_chain']
            pred_chains = r['sorted_pred_chains']
            pred_chains_flat = [s for b in pred_chains[:k] for s in b]
            TP += num_TP_sp(pred_chains_flat, sp_list)
            NP += len(set(pred_chains_flat))
            NT += len(sp_list)
        prec = TP / NP;
        recl = TP / NT;
        f1 = 2 * prec * recl / (prec + recl)
        print("sorted union prediction f1 wrt %s set:" % args.chain_f1_target)
        print("Prec:", prec, "Recl:", recl, "F1:", f1)

    '''
    # chain length
    lengths = []
    for r in res:
        lengths.append([len(chain) for chain in r['pred_chains'][:k]])
    print("top k chain avg min length:", sum([min(ls) for ls in lengths]) / len(lengths))
    print("top k chain avg max length:", sum([max(ls) for ls in lengths]) / len(lengths))
    print("top k chain avg avg length:", sum([sum(ls) / len(ls) for ls in lengths]) / len(lengths))

    if 'sent_labels' in res[0]:
        # sorted chain length
        lengths = []
        for r in res:
            pred_chains = r['sorted_pred_chains']
            lengths.append([len(chain) for chain in pred_chains[:k]])
        print("top k sorted chain avg min length:", sum([min(ls) for ls in lengths]) / len(lengths))
        print("top k sorted chain avg max length:", sum([max(ls) for ls in lengths]) / len(lengths))
        print("top k sorted chain avg avg length:", sum([sum(ls) / len(ls) for ls in lengths]) / len(lengths))

    if 'possible_chain' in res[0]:
        # length of possible chain
        ls = [len(r['possible_chain']) for r in res]
        ls_g = [len(r['possible_chain']) for r in res if len(r['possible_chain']) > 0]
        print("length of rule-based", sum(ls) / len(ls))
        print("length of non-zero rule-based", sum(ls_g) / len(ls_g))
        print("number of zero rule-based", len(ls) - len(ls_g))

    if 'sent_labels' in res[0]:
        # length of sp set
        ls = [len([i for i, ind in enumerate(r['sent_labels']) if ind == 1]) for r in res]
        ls_g = [l for l in ls if l > 0]
        print("length of sp", sum(ls) / len(ls))
        print("length of non-zero sp", sum(ls_g) / len(ls_g))
        print("number of zero sp", len(ls) - len(ls_g))
    '''

    # chain union avg length
    NP = 0
    for r in res:
        pred_chains_flat = [s for b in r['pred_chains'][:k] for s in b]
        NP += len(set(pred_chains_flat))
    print("top k chain union avg length:", NP / len(res))

    if 'sent_labels' in res[0]:
        # f1 between the union of topk best TP chain and sp set
        NP = 0
        for r in res:
            pred_chains = r['sorted_pred_chains']
            pred_chains_flat = [s for b in pred_chains[:k] for s in b]
            NP += len(set(pred_chains_flat))
        print("top k sorted chain union avg length:", NP / len(res))

