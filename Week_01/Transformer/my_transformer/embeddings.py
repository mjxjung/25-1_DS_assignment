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
        self.d_model = d_model # 단어 벡터의 차원 수
        pe = torch.zeros(max_len, d_model)
        # 단어 순서 (pos), 내장 벡터의 차원위치(i)에 의해 고유하게 정해지는 값의 표
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))        
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
                
        self.pe = pe.unsqueeze(0) # pe 앞에 미니배치차원 (1, max_len, d_model)
        # 위치 인코딩은 학습되는 파라미터가 아니므로 고정
        self.pe.requires_grad = False        
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return math.sqrt(self.d_model)*x + self.pe[:, :x.size(1), :]