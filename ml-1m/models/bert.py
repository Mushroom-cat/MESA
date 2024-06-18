from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn
import torch


class BERTModel(BaseModel):
    def __init__(self, inter_mat, usernum, itemnum, args):
        super().__init__(args)
        self.bert = BERT(inter_mat, usernum, itemnum, args)
        #self.out = nn.Linear(self.bert.hidden, args.num_items + 1)
        self.item_emb = nn.Embedding(args.num_items + 1, args.bert_hidden_units, padding_idx=0)
        self.dev = args.device

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, user_id, x, pos_seqs, neg_seqs):

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        log_feats, balance_loss = self.bert(user_id, x)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, balance_loss

    def predict(self, user_id, log_seqs, item_indices):  # for inference

        log_seqs = torch.tensor(log_seqs).unsqueeze(0)
        log_feats, _ = self.bert(user_id, log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
