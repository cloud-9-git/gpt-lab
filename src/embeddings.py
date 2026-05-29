# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length

        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        # nn.Embedding: 임베딩 층을 랜덤으로 초기화해서 객체를 만들어줌.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        # shape[0] = 행 크기 반환 (batch_size), shape[1] = 열 크기 반환 (seq_len)
        seq_len = x.shape[1]
        # 임베딩 벡터에 넣을 벡터 크기
        positions = torch.arange(seq_len)

        # 행렬 모양이 달라보이지만 파이토치가 자동으로 pos_emb에 tok_emb 배치를 복사해서 더해줌으로써 차원 맞춤 (브로드캐스팅)
        tok_emb = self.token_embedding(x)            # [batch_size, seq_len, emb_dim]
        pos_emb = self.position_embedding(positions) # [seq_len, emb_dim]

        # 토큰 임베딩값과 위치 임베딩값 더한거 드롭아웃 적용하고 반환함
        return self.dropout(tok_emb + pos_emb)


