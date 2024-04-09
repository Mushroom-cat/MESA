import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

import scipy.sparse as sp


class LightGCN(nn.Module):

    def __init__(self, inter_mat, user_num, item_num, args):
        super(LightGCN, self).__init__()

        # load dataset info
        self.interaction_matrix = inter_mat.astype(np.float32)

        # load parameters info
        self.latent_dim = args.embedding_dim
        self.n_layers = args.gcn_layers
        self.n_users = user_num+1
        self.n_items = item_num+1

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(args.device)

        # parameters initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):

        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, u):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return F.embedding(u, user_all_embeddings)

    # def predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #
    #     user_all_embeddings, item_all_embeddings = self.forward()
    #
    #     u_embeddings = user_all_embeddings[user]
    #     i_embeddings = item_all_embeddings[item]
    #     scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
    #     return scores
    #
    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     if self.restore_user_e is None or self.restore_item_e is None:
    #         self.restore_user_e, self.restore_item_e = self.forward()
    #     # get user embedding from storage variable
    #     u_embeddings = self.restore_user_e[user]
    #
    #     # dot with all item embedding to accelerate
    #     scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
    #
    #     return scores.view(-1)




class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs



class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, modulation_mode, dropout=0.1, original_mode=False):
        super().__init__()
        assert d_model % h == 0

        self.original_mode = original_mode

        self.d_k = d_model // h
        self.h = h

        if self.original_mode:
            self.linear_layers = nn.ModuleList(
                [nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)])
        else:
            assert (modulation_mode == 'scaling') or (modulation_mode == 'shifting')
            if modulation_mode == 'shifting':
                self.linear_layers = nn.ModuleList([LinearForMeta_sh(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)])
            elif modulation_mode == 'scaling':
                self.linear_layers = nn.ModuleList([LinearForMeta_sc(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)])

        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, fc_w=None, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        if self.original_mode:
            query = self.linear_layers[0](query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query = self.linear_layers[0](query, fc_w).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_layers[1](key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_layers[2](value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)


        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class OriginalMultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, d_output, dropout=0.1, max_len=50, args=None):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)])
        self.output_linear = nn.Linear(d_model, d_output)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.linear_layers[0](query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_layers[1](key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_layers[2](value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

# shifting mode

class LinearForMeta_sh(nn.Module):

    def __init__(self, in_dim, out_dim, lam=0.01):
        super().__init__()

        self.base_w = Parameter(torch.zeros(in_dim, out_dim))
        self.b = Parameter(torch.zeros(out_dim))
        self.lam = lam
        nn.init.normal_(self.b, std=0.01)
        nn.init.xavier_normal_(self.base_w)

    def forward(self, x, w):

        w1 = torch.mul(w, self.base_w)
        x1 = torch.matmul(x, w1)
        x2 = x1 + self.b

        return x2


# scaling mode

class LinearForMeta_sc(nn.Module):

    def __init__(self, in_dim, out_dim, lam=0.01):
        super().__init__()

        self.base_w = Parameter(torch.zeros(in_dim, out_dim))
        self.b = Parameter(torch.zeros(out_dim))
        self.lam = lam
        nn.init.normal_(self.b, std=0.01)
        nn.init.xavier_normal_(self.base_w)

    def forward(self, x, w):

        w1 = self.lam * w + self.base_w
        x1 = torch.matmul(x, w1)
        x2 = x1 + self.b

        return x2


class MetaNetwork(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim0, out_dim1):
        super().__init__()

        # self.norm1 = nn.BatchNorm1d(in_dim)
        # self.norm2 = nn.BatchNorm1d(hidden_dim)

        self.out_dim0 = out_dim0
        self.out_dim1 = out_dim1
        self.activation = nn.Sigmoid()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim0 * out_dim1)

        # for name, param in self.named_parameters():
        #     try:
        #         torch.nn.init.xavier_normal_(param.data)
        #     except:
        #         pass
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, user_g_emb):

        # x1 = self.fc1(self.norm1(user_attr_emb))
        # x2 = self.fc2(self.norm2(user_attr_emb))

        x1 = self.activation(self.fc1(user_g_emb))
        x2 = self.fc2(x1)

        return x2.view(-1, self.out_dim0, self.out_dim1)


class user_embedding(torch.nn.Module):
    def __init__(self, config):
        super(user_embedding, self).__init__()
        self.num_gender = config.num_gender
        self.num_age = config.num_age
        self.num_occupation = config.num_occupation
        self.num_zipcode = config.num_zipcode
        self.embedding_dim = config.embedding_dim

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)


        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)

class MetaSANRec(torch.nn.Module):
    def __init__(self, inter_mat, user_num, item_num, args):
        super(MetaSANRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.num_blocks = args.num_blocks
        self.moe_num = args.moe_num
        self.disable_moe = args.disable_moe

        self.original_mode = args.original_mode

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.gcn_layers = torch.nn.ModuleList()
        self.meta_layers = torch.nn.ModuleList()
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.user_emb = user_embedding(args)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)


        for _ in range(args.num_blocks):

            if self.disable_moe:
                new_gnc_layer = LightGCN(inter_mat, user_num, item_num, args)
                self.gcn_layers.append(new_gnc_layer)
            else:
                for __ in range(args.moe_num):
                    new_gnc_layer = LightGCN(inter_mat, user_num, item_num, args)
                    self.gcn_layers.append(new_gnc_layer)
                self.gating_network = OriginalMultiHeadedAttention(h=args.num_heads, d_model=args.hidden_units, d_output=self.moe_num,
                                                                   dropout=args.dropout_rate)

            new_meta_layer = MetaNetwork(args.embedding_dim, args.meta_hidden_units, args.hidden_units, args.hidden_units)
            self.meta_layers.append(new_meta_layer)

            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
            #                                                 args.num_heads,
            #                                                 args.dropout_rate)
            new_attn_layer = MultiHeadedAttention(args.num_heads,args.hidden_units,args.modulation_mode,args.dropout_rate,original_mode=self.original_mode)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, user_id, log_seqs):

        #logseqs:128, 200
        #mask:128, 1, 200, 200
        attention_mask = torch.tensor((log_seqs > 0), device=self.dev).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1)

        seqs = self.item_emb(torch.tensor(log_seqs, dtype=torch.long).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        #tl = seqs.shape[1] # time dim len for enforce causality
        #attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        balance_loss_list = []
        for i in range(len(self.attention_layers)):

            if self.original_mode:
                fc_w = None
            else:
                if self.disable_moe:
                    user_g_emb = self.gcn_layers[i](user_id)
                else:
                    all_user_g_emb = []
                    for j in range(self.moe_num):
                        all_user_g_emb.append(self.gcn_layers[i * self.num_blocks + j](user_id))

                    #new: gating
                    moe_weights = self.gating_network.forward(seqs, seqs, seqs, mask=attention_mask)

                    moe_weights = moe_weights[:, -1, :]
                    moe_weights = F.softmax(moe_weights, dim=-1)
                    average_weights = torch.mean(moe_weights, dim=0)
                    balance_loss = torch.std(average_weights)
                    balance_loss_list.append(balance_loss)

                    moe_weights = moe_weights.T.unsqueeze(-1)  # (3,b,1)

                    stacked_user_g_emb = torch.stack(all_user_g_emb, dim=0)
                    if stacked_user_g_emb.dim() == 2:
                        stacked_user_g_emb = stacked_user_g_emb.unsqueeze(1)

                    user_g_emb = torch.mul(moe_weights, stacked_user_g_emb)
                    user_g_emb = torch.mean(user_g_emb, dim=0)
                    #new: gating

                fc_w = self.meta_layers[i](user_g_emb)

            #seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs, seqs, fc_w, mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            #seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        final_balance_loss = torch.mean(torch.stack(balance_loss_list))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, final_balance_loss

    def forward(self, user_id, log_seqs, pos_seqs, neg_seqs): # for training


        log_feats, balance_loss = self.log2feats(user_id, log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, balance_loss


    def predict(self, user_id, log_seqs, item_indices): # for inference

        log_seqs = torch.tensor(log_seqs).unsqueeze(0)

        log_feats, _ = self.log2feats(user_id, log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
