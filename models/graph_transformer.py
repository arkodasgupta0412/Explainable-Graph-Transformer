import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import EfficientMultiHeadAttention

class GraphTransformerNet(nn.Module):

    """
    Graph Transformer network combining multi-head attention, residuals,
    feed-forward layers, and positional encodings.
    """
    
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, n_classes, pos_enc_dim=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.pos_encoder = nn.Linear(pos_enc_dim, hidden_dim) if pos_enc_dim > 0 else None
        self.dropout = dropout
        
        self.layers = nn.ModuleList([
            self._build_layer(hidden_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, n_classes)
        

    def _build_layer(self, hidden_dim, num_heads, dropout):
        return nn.ModuleDict({
            'attention': EfficientMultiHeadAttention(hidden_dim, hidden_dim, num_heads),
            'norm1': nn.LayerNorm(hidden_dim),
            'ffn': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ),
            'norm2': nn.LayerNorm(hidden_dim)
        })
    
    
    def forward(self, data):
        x = self.embedding(data.x)
        if self.pos_encoder is not None and hasattr(data, 'lap_pos_enc'):
            x = x + self.pos_encoder(data.lap_pos_enc)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for layer in self.layers:
            attn_out = layer['attention'](x, data.edge_index)
            attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
            x = layer['norm1'](x + attn_out)
            
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        return self.classifier(x)
