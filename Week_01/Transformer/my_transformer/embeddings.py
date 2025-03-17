import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #TODO
        # 학습 가능한 위치 임베딩 레이어 생성
        self.position_embedding = nn.Embedding(max_len, d_model) #(max_len, d_model) 사이즈로 학습가능한 임베딩
        pe = torch.zeros(max_len, d_model) # 초기화 필요

        pos_term = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos_term * div_term) # 인덱스 2i
        pe[:, 1::2] = torch.cos(pos_term * div_term) # 인덱스 2i+1

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.position_embedding(x) + self.pe[:, :x.size(1), :]


        
        
  
