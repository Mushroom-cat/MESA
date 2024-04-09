# Introduction

This is our PyTorch implementation for the paper:

**MESA: Meta-model Enhanced Self-Attentive Sequential Recommendation**

It includes the codes for SASRec+MESA on two datasets (Movielens-1M and Dbook). You can change 'model.py' to use other base models.

Please cite our paper if you use the code.

The codes are based on the PyTorch implementation of SASRec. (https://github.com/pmixer/SASRec.pytorch)

# Environment

PyTorch 1.12.0

Python 3.9.7

# Datasets

The preprocessed datasets are included in the repo (ml-1m/data/ml-1m, dbook/data/dbook)

# Model Training & Testing

You can change the --modulation_mode argument between 'shifting' and 'scaling' to use different modulation methods of MESA.

### Movielens-1M

First, enter 'ml-1m' directory.

Training:

`python main.py --dataset=ml-1m --train_dir=default --device cuda --modulation_mode shifting`

The trained model will be saved as 'ml-1m_default/?.pth'. Find its name and replace the '?'.

Testing:

`python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/?.pth' --inference_only=true --modulation_mode shifting`

### Dbook

First, enter 'dbook' directory.

Training:

`python main.py --dataset=dbook --train_dir=default --device cuda --modulation_mode shifting`

The trained model will be saved as 'dbook_default/?.pth'. Find its name and replace the '?'.

Testing:

`python main.py --device=cuda --dataset=dbook --train_dir=default --state_dict_path='dbook_default/?.pth' --inference_only=true --modulation_mode shifting`
