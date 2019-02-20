import json

data_number = 2
start = 0
original_path = '/backup2/jfchen/data/hotpot/hotpot_train_v1.json'
output_url = '/backup2/jfchen/data/hotpot/hotpot_test.json'

original_data = json.load(open(original_path, 'r'))

test_data = []
count = 0
file_count = 0

for i in range(start, start+data_number):
    count += 1
    test_data.append(original_data[i])

print(len(test_data))
json.dump(test_data, open(output_url, 'w'), sort_keys=True, indent=4)

# from typing import List
#
# test =[1,2,3,4,5]
#
# def generate_all_permutations(original: List[int], res: List[int]):
