import torch.nn as nn
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F

from .attention import MultiHeadedAttention, OriginalMultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward

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

class MoeGatingLayer(nn.Module):

    def __init__(self, n_expert, ):
        super().__init__()

        self.out_dim0 = out_dim0
        self.out_dim1 = out_dim1
        self.activation = nn.Sigmoid()

        self.attention = OriginalMultiHeadedAttention(h=attn_heads, d_model=hidden, d_output=n_expert, dropout=att_dropout, max_len=max_len,
                                              args=args)
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

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, inter_mat, user_num, item_num, hidden, attn_heads, feed_forward_hidden, dropout, att_dropout=0.2, residual=True, activate="gelu", max_len=50, args=None):
        super().__init__()

        #self.gcn_layer = LightGCN(inter_mat, user_num, item_num, args)

        self.moe_num = args.moe_num
        self.gcn_layers = torch.nn.ModuleList()
        for __ in range(self.moe_num):
            new_gnc_layer = LightGCN(inter_mat, user_num, item_num, args)
            self.gcn_layers.append(new_gnc_layer)

        self.gating_network = OriginalMultiHeadedAttention(h=attn_heads, d_model=hidden, d_output=self.moe_num,
                                                      dropout=att_dropout, max_len=max_len,
                                                      args=args)

        self.meta_layer = MetaNetwork(args.embedding_dim, args.meta_hidden_units, hidden, hidden)
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=att_dropout, max_len=max_len, args=args)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, activate=activate)

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.residual = residual

    def forward(self, x, mask, user_id=None, stride=None, args=None, users=None):

        all_user_g_emb = []
        for j in range(self.moe_num):
            all_user_g_emb.append(self.gcn_layers[j](user_id))

        moe_weights = self.gating_network.forward(x, x, x, mask=mask, stride=stride, args=args, users=users)

        moe_weights = moe_weights[:, -1, :]  #also try .sum(dim=1)??   (b, 3)
        moe_weights = F.softmax(moe_weights, dim=-1)
        average_weights = torch.mean(moe_weights, dim=0)
        balance_loss = torch.std(average_weights)

        moe_weights = moe_weights.T.unsqueeze(-1) # (3,b,1)

        stacked_user_g_emb = torch.stack(all_user_g_emb, dim=0)
        if stacked_user_g_emb.dim() == 2:
            stacked_user_g_emb = stacked_user_g_emb.unsqueeze(1)

        user_g_emb = torch.mul(moe_weights, stacked_user_g_emb)
        user_g_emb = torch.mean(user_g_emb, dim=0)  #3,32

        fc_w = self.meta_layer(user_g_emb)

        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, fc_w, mask=mask, stride=stride, args=args, users=users))
        x = self.output_sublayer(x, self.feed_forward)
        return x, balance_loss