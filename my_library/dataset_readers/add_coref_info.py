import json, pickle
import argparse
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge the hotpot data with coref info')
    parser.add_argument('data_path', help='path to hotpot json data file')
    parser.add_argument('coref_path', help='path to the coref info')
    parser.add_argument('output_path', help='path to store the merged json data file')
    parser.add_argument('--bypara', action='store_true', default=False, help='whether the coref info is processed in para basis')
    parser.add_argument('--key', default='coref_clusters', help='the key to use to add in the hotpot dataset')
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)
    with open(args.coref_path, 'rb') as f:
        coref = pickle.load(f)

    for instance, coref_instance in tqdm(zip(data, coref)):
        assert instance['_id'] == coref_instance['_id']
        if args.bypara:
            coref_doc = []
            coref_clusters = []
            for title, para_coref in coref_instance['coref_info']:
                offset = len(coref_doc)
                for c in para_coref['clusters']:
                    shifted_c = [[s+offset, e+offset] for s, e in c]
                    coref_clusters.append(shifted_c)
                coref_doc.extend(para_coref['document'])
        else:
            info = coref_instance['coref_info']
            if not info is None:
                coref_clusters = info['clusters']
            else:
                coref_clusters = []
        instance[args.key] = coref_clusters
    with open(args.output_path, 'w') as f:
        json.dump(data, f)
