import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO
        head_dim = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 마스킹 적용
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, float('-inf'))
        
        # 소프트맥스로 가중치 계산
        attn_weights = F.softmax(attn_scores, dim=-1) 
        
        # 어텐션 가중치
        output = torch.matmul(attn_weights, v) 
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        # batch_size = Q.size(0)
        # seq_len = Q.size(1)

        # # 마스크 차원 확장
        # if mask is not None:
        #     mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1) 
        batch_size, seq_len, _ = Q.shape  # `Q.size(0)` 대신 `Q.shape` 활용 (가독성 개선)

        # 마스크 차원 확장
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # `expand` 대신 `repeat` 사용 (동일한 효과)


        print(f"Q.shape: {Q.shape}")  # 원본 Q 크기 확인
        print(f"Q_proj.shape before reshape: {self.query_layers(Q).shape}")  # 변환 전 크기 확인


        # Query, Key, Value의 projection 수행 및 차원 변형
        Q_proj = self.query_layers(Q).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2).contiguous()
        K_proj = self.key_layers(K).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2).contiguous()
        V_proj = self.value_layers(V).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2).contiguous()

        # Scaled Dot-Product 어텐션 적용
        attn_output, attn_weights = self.attn(Q_proj, K_proj, V_proj, mask)

        # 멀티헤드 결합 (Transpose 후 Reshape)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        # 최종 선형 계층 적용
        output = self.fc(attn_output)

        return output