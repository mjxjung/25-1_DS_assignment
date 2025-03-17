import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward_layer = FeedForwardLayer(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.dropout_layer1 = DropoutLayer(dropout)
        self.dropout_layer2 = DropoutLayer(dropout)
        self.residual_conn1 = ResidualConnection()
        self.residual_conn2 = ResidualConnection()
    
    def forward(self, x: Tensor) -> Tensor:
        mask = None
        #TODO
        attention_output = self.residual_conn1(x, lambda x: self.self_attention(x, x, x, mask))
        attention_output = self.layer_norm1(attention_output)
        
        # feed-forward 
        feedforward_output = self.residual_conn2(attention_output, self.feed_forward_layer)
        feedforward_output = self.layer_norm2(feedforward_output)
        
        return feedforward_output
