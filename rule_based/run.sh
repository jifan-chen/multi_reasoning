#!/usr/bin/env bash
for index in 3 4 5 6 7 8 9
do
#    index=3
    data_path="/scratch/cluster/jfchen/jfchen/data/hotpot/train_pred_chain/pred_train${index}.json"
    output_path="/scratch/cluster/jfchen/jfchen/data/hotpot/train_full_data_oracle_overlap/train${index}.json"
    get_chain="overlap"
    task="hotpot"

    command="python rule_based.py --data_path ${data_path} --output_path ${output_path} --get_chain ${get_chain}
              --task ${task} &"

    echo ${command}
    eval ${command}
done