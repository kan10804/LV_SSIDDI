import torch
from torch import nn
import torch.nn.functional as F

# Single-Head Co-Attention Layer
class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        return e_scores


# Multi-Head Co-Attention 
class MultiCoAttentionLayer(nn.Module):
    def __init__(self, n_features, n_heads=4):
        super().__init__()
        self.n_features = n_features
        self.n_heads = n_heads
        hidden_dim = n_features // 2

        self.w_q = nn.Parameter(torch.zeros(n_heads, n_features, hidden_dim))
        self.w_k = nn.Parameter(torch.zeros(n_heads, n_features, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(n_heads, hidden_dim))
        self.a = nn.Parameter(torch.zeros(n_heads, hidden_dim))
        self.alpha = nn.Parameter(torch.ones(n_heads))  # weight for each head

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))

    def forward(self, receiver, attendant):
        attn_heads = []

        for i in range(self.n_heads):
            keys = receiver @ self.w_k[i]
            queries = attendant @ self.w_q[i]
            bias = self.bias[i].view(1,1,1,-1)
            e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + bias
            e_scores = torch.tanh(e_activations) @ self.a[i]
            attn_heads.append(e_scores)

        attn_heads = torch.stack(attn_heads)
        weights = torch.softmax(self.alpha, dim=0)

        attentions = torch.zeros_like(attn_heads[0])
        for i in range(self.n_heads):
            attentions += weights[i] * attn_heads[i]

        attentions = torch.softmax(attentions, dim=-1)
        return attentions


# RESCAL
class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores=None):
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        rels = self.rel_emb(rels)
        rels = F.normalize(rels, dim=-1)
        rels = rels.view(-1, self.n_features, self.n_features)
        scores = heads @ rels @ tails.transpose(-2, -1)
        if alpha_scores is not None:
            scores = alpha_scores * scores
        return scores.sum(dim=(-2, -1))
