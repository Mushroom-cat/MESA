from templates import set_template
from models import MODELS

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
parser.add_argument('--template', type=str, default=None)

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)


################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=None, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=None, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')
parser.add_argument('--bert_att_dropout', type=float, default=0.2, help='Dropout probability to use throughout the attention scores')
# DAE #
parser.add_argument('--dae_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--dae_num_hidden', type=int, default=0, help='Number of hidden layers in DAE')
parser.add_argument('--dae_hidden_dim', type=int, default=600, help='Dimension of hidden layer in DAE')
parser.add_argument('--dae_latent_dim', type=int, default=200, help="Dimension of latent vector in DAE")
parser.add_argument('--dae_dropout', type=float, default=0.5, help='Probability of input dropout in DAE')
# VAE #
parser.add_argument('--vae_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--vae_num_hidden', type=int, default=0, help='Number of hidden layers in VAE')
parser.add_argument('--vae_hidden_dim', type=int, default=600, help='Dimension of hidden layer in VAE')
parser.add_argument('--vae_latent_dim', type=int, default=200, help="Dimension of latent vector in VAE (K in paper)")
parser.add_argument('--vae_dropout', type=float, default=0.5, help='Probability of input dropout in VAE')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')


#####
#ex


parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--meta_lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--meta_hidden_units', default=100, type=int)

parser.add_argument('--num_items', default=22347, type=int)

parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=bool)
parser.add_argument('--state_dict_path', default=None, type=str)


parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--gcn_layers', type=int, default=3)
parser.add_argument('--moe_num', type=int, default=3)
parser.add_argument('--num_gender', type=int, default=2)
parser.add_argument('--num_age', type=int, default=7)
parser.add_argument('--num_occupation', type=int, default=21)
parser.add_argument('--num_zipcode', type=int, default=3402)
parser.add_argument('--existing_split', type=float, default=0.7)
parser.add_argument('--balance_loss_weight', type=float, default=10)

parser.add_argument('--local_type', type=str, default='none', help='Types of localness bias')
parser.add_argument('--local_num_heads', type=int, default=3, help='Local att heads num')

parser.add_argument('--least_inter_num', type=int, default=3)
parser.add_argument('--most_inter_num', type=int, default=50)
parser.add_argument('--modulation_mode', default='shifting', type=str)
################
args = parser.parse_args()
set_template(args)
