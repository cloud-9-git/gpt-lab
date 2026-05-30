# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        
        # qkv projection: 입력 x를 query/key/value 표현으로 바꾸기 위한 선형변환
        # 예: nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        # 왜? attention은 Q, K, V 세 표현을 기준으로 계산되기 때문.

        # output projection: 여러 head 결과를 다시 d_model 차원으로 정리하는 선형변환
        # 왜? head를 합친 뒤 다음 블록이 받을 수 있는 형태로 맞춰야 하기 때문.

        # dropout: attention weight에 적용할 regularization 정의
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.drop_out = nn.Dropout(drop_rate)
        
        raise NotImplementedError("MultiHeadAttention.__init__을 구현하세요.")

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        
        # 1. x를 선형변환해서 Q, K, V를 한 번에 만든다.
        # shape: (B, T, 3 * d_model)
        # 왜? 입력을 attention 계산용 세 표현(query, key, value)으로 바꿔야 하고,
        # 한 번의 projection으로 Q/K/V를 함께 계산하면 효율적이고 구현도 단순하기 때문.
        qkv = self.qkv_proj(x)

        # 2. qkv를 Q, K, V로 나눈다.
        # qkv.chunk(3, dim=-1) 사용.
        # shape: q, k, v 각각 (B, T, d_model)
        # 왜? Q, K, V는 이후 역할이 다르므로 분리해서 따로 계산해야 하기 때문.
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. 각 텐서를 multi-head 형태로 바꾼다.
        # (B, T, d_model) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        # 왜? 여러 head가 서로 다른 관점으로 attention을 보게 하려면
        # d_model을 head 단위로 나눠서 병렬 계산해야 하기 때문.

        # 4. attention score를 계산한다.
        # scores = q @ k.transpose(-2, -1)
        # shape: (B, n_heads, T, T)
        # 그리고 sqrt(head_dim)으로 나눠 scaling 한다.
        # 왜? 각 query가 모든 key와 얼마나 관련 있는지 점수로 계산해야 하며,
        # scaling을 하지 않으면 score 분산이 커져 softmax가 과도하게 뾰족해지고
        # gradient가 불안정해질 수 있기 때문.

        # 5. causal mask를 적용한다.
        # causal_mask=True이면 미래 토큰 위치를 보지 못하게 mask 처리한다.
        # (T, T) upper-triangular mask를 diagonal=1로 만들고, 가릴 위치를 -inf로 채운다.
        # mask는 x.device 위에 만들고, (B, n_heads, T, T)에 broadcast되게 둔다.
        # 왜? GPT는 다음 토큰 예측 모델이라 현재 위치가 미래 정보를 보면 안 되기 때문.

        # 6. softmax로 attention weight를 만든다.
        # attn_weights = torch.softmax(scores, dim=-1)
        # 정의한 dropout은 일반적으로 attention weight에 적용한다.
        # 왜? score를 확률처럼 해석 가능한 가중치로 바꿔야 어떤 토큰을 얼마나 참고할지 정할 수 있기 때문.

        # 7. attention weight와 V를 곱해 context를 만든다.
        # context = attn_weights @ v
        # shape: (B, n_heads, T, head_dim)
        # 왜? 각 위치의 출력 벡터를, 참고할 토큰들의 value를 가중합해서 만들기 때문.

        # 8. 여러 head를 다시 합친다.
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, d_model)
        # transpose 뒤에는 contiguous().view(...) 또는 reshape(...)로 메모리 배치를 맞춘다.
        # 왜? head별로 나눠 계산한 결과를 다시 원래 hidden 차원으로 합쳐야 다음 레이어가 같은 형식의 입력을 받을 수 있기 때문.

        # 9. output projection을 적용한다.
        # out = self.out_proj(context)
        # output dropout을 별도로 둘 수도 있지만, 이 템플릿에서는 attention weight dropout만으로도 충분하다.
        # 왜? 합쳐진 head 정보를 다시 모델 차원으로 정리해서 다음 블록으로 넘기기 좋은 표현으로 만들기 때문.

        # 10. 반환 형식을 처리한다.
        # return_attention_weights=False면 out만 반환한다.
        # True면 (out, attn_weights)를 반환한다.
        # 왜? 기본적으로는 attention 출력만 있으면 되지만, 디버깅이나 시각화를 위해 attention weight도 확인할 수 있어야 하기 때문.
        
        raise NotImplementedError("MultiHeadAttention.forward를 구현하세요.")
