import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import GINConv, SAGPooling, LayerNorm, global_add_pool

from layers import CoAttentionLayer, MultiCoAttentionLayer, RESCAL

class SSI_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total,
                 heads_out_feat_params, blocks_params):

        super().__init__()

        self.initial_norm = LayerNorm(in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        self.n_blocks = len(blocks_params)
        self.kge_dim = kge_dim

        # GIN Blocks 
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = SSI_DDI_Block(n_heads, in_features, head_out_feats, hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        # Co-Attention Layers
        self.co_single = CoAttentionLayer(kge_dim)
        self.co_multi = MultiCoAttentionLayer(kge_dim, n_heads=4)

        # RESCAL
        self.KGE = RESCAL(rel_total, kge_dim)

    def forward(self, triples):
        h_data, t_data, rels = triples
        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

        repr_h, repr_t = [], []

        for i, block in enumerate(self.blocks):
            h_data, r_h = block(h_data)
            t_data, r_t = block(t_data)
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.gelu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.gelu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        # Combie single and multi-head co-attention
        att_single = self.co_single(repr_h, repr_t)
        att_multi = self.co_multi(repr_h, repr_t)
        alpha = torch.sigmoid(att_single)
        attn_combined = alpha * att_multi + (1 - alpha) * att_single
        scores = self.KGE(repr_h, repr_t, rels, attn_combined)
        return scores


class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        hidden_dim = head_out_feats * n_heads

        mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINConv(mlp)
        self.pool = SAGPooling(hidden_dim, min_score=-1)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        x, edge_index, _, batch, _, _ = self.pool(data.x, data.edge_index, batch=data.batch)
        graph_emb = global_add_pool(x, batch)
        data.x = x
        return data, graph_emb
