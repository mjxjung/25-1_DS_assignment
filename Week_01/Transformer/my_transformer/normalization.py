import torch.nn as nn
from torch import Tensor

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(LayerNormalization, self).__init__()
        #TODO one line!
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)  # LayerNorm의 `eps` 값을 변경하여 안정성 향상
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.norm(x.contiguous())  # contiguous로 사용하여 메모리 연속성 보장