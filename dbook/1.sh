#!/bin/bash

for((i=11;i<=41;i+=10));
do
python main.py --device=cuda --dataset=dbook --train_dir=default --state_dict_path='dbook_default/bias10.pth' --inference_only=true --maxlen=200 --template train_bert --least_inter_num $i --most_inter_num $(expr $i + 10);
done