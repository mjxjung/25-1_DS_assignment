import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        #TODO
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward_layer = FeedForwardLayer(d_model, d_ff)
        self.norm_layer1 = LayerNormalization(d_model)
        self.norm_layer2 = LayerNormalization(d_model)
        self.norm_layer3 = LayerNormalization(d_model)
        self.residual_conn1 = ResidualConnection()
        self.residual_conn2 = ResidualConnection()
        self.residual_conn3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        x = self.residual_conn1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.norm_layer1(x)
        
        # Cross-attention 
        x = self.residual_conn2(x, lambda x: self.encoder_decoder_attention(x, memory, memory, src_mask))
        x = self.norm_layer2(x)
        
        # Feedforward 
        x = self.residual_conn3(x, self.feed_forward_layer)
        x = self.norm_layer3(x)

        return x