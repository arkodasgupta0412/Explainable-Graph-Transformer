import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class EfficientMultiHeadAttention(MessagePassing):

    """
    Memory-efficient multi-head attention layer for graph data.
    Uses message passing to compute attention scores and propagate messages.
    """

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__(aggr='add', node_dim=0)
        self.head_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        

    def forward(self, x, edge_index):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        out = self.propagate(edge_index, q=q, k=k, v=v)
        return self.out_proj(out.view(-1, self.num_heads * self.head_dim))
    
    
    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        score = (q_i * k_j).sum(dim=-1) * self.scale
        score = softmax(score, index, ptr, size_i)
        return v_j * score.unsqueeze(-1)
