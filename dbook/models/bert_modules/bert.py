from torch import nn as nn
import torch

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, inter_mat, user_num, item_num, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        self.args = args
        max_len = args.bert_max_len
        num_items = args.num_items
        self.n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        att_dropout = args.bert_att_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        self.transformer_blocks = nn.ModuleList()
        for _ in range(self.n_layers):

            new_trans_block = TransformerBlock(inter_mat, user_num, item_num, hidden, heads, hidden * 4, dropout, att_dropout, max_len=max_len, args=args).to(args.device)
            self.transformer_blocks.append(new_trans_block)

        #self.user_emb = user_embedding(args)

    def forward(self, user_id, x):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        # for transformer in self.transformer_blocks:
        #     x = transformer.forward(x, mask)

        # ???

        balance_loss_list = []
        for transformer in self.transformer_blocks:
            x, balance_loss = transformer(x, mask, user_id, args=self.args)
            balance_loss_list.append(balance_loss)
        final_balance_loss = torch.mean(torch.stack(balance_loss_list))

        return x, final_balance_loss

    def init_weights(self):
        pass
