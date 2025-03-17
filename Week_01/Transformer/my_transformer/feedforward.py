import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForwardLayer, self).__init__()
        #TODO two lines!
        self.hidden_layer = nn.Linear(d_model, d_ff)  # 첫 번째 선형 변환 (입력 차원 → 확장된 차원)
        self.output_layer = nn.Linear(d_ff, d_model)  # 두 번째 선형 변환 (확장된 차원 → 원래 차원)
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.output_layer(F.gelu(self.hidden_layer(x)))  # 활성화 함수를 GELU로 적용

class DropoutLayer(nn.Module):
    def __init__(self, p: float) -> None:
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

class ActivationLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return torch.tanh(x)  # Tanh 활성화 함수 적용



