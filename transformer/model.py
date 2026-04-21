import torch
import torch.nn as nn
import math

from . import _model_stack


def create_causal_mask(size):
    """Create causal mask to prevent attending to future tokens"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None, is_causal=False):
        batch_size = Q.size(0)

        q_proj, k_proj, v_proj = _model_stack.runtime_qkv(self.W_q, self.W_k, self.W_v, Q, K, V)
        Q = q_proj.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = k_proj.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = v_proj.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output = _model_stack.runtime_attention(Q, K, V, mask=mask, is_causal=is_causal)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return _model_stack.runtime_linear(self.W_o, output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, is_causal=False):
        attention_output = self.attention(x, x, x, mask=mask, is_causal=is_causal)
        x = _model_stack.runtime_add_layer_norm(x, self.dropout(attention_output), self.norm1)
        ff_output = _model_stack.runtime_mlp(self.ff_in, self.ff_out, x, activation="relu")
        x = _model_stack.runtime_add_layer_norm(x, self.dropout(ff_output), self.norm2)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        use_causal = mask is None

        x = _model_stack.runtime_embedding(self.embedding, x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask, is_causal=use_causal)

        return _model_stack.runtime_linear(self.fc, x)
