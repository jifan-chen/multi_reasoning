#!/usr/bin/env bash
for index in 0 1 2 3 4 5 6 7 8 9
do
#    index=3
    original_path="/scratch/cluster/jfchen/jfchen/data/wikihop/train_ner/train${index}.json"
    output_path="/scratch/cluster/jfchen/jfchen/data/wikihop/train_selected/train${index}.json"
    task="wikihop"

    command="python generate_selected_data.py --original_path ${original_path} --output_path ${output_path}
              --task ${task} &"

    echo ${command}
    eval ${command}
done

exit 1