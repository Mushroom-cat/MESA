This is our PyTorch implementation for the paper:
？？？

这些代码是基于SASRec's PyTorch implementation（超链接）的。

python main.py --dataset=ml-1m --train_dir=default --device cuda --modulation_mode shifting


python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/sc.pth' --inference_only=true --modulation_mode scaling