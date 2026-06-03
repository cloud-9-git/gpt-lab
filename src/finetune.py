# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_tsv(path: str | Path | None) -> list[dict]:
        if path is None:
            return []

        rows = []
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            next(f, None)  # header skip
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue

                text = parts[1].strip()
                label = parts[2].strip()

                if not text or not label:
                    continue

                rows.append({
                    "text": text,
                    "label": int(label),
                })

        return rows

    train_data = read_tsv(train_tsv_path)
    test_data = read_tsv(test_tsv_path)

    rng = random.Random(seed)
    rng.shuffle(train_data)

    val_size = int(len(train_data) * val_ratio)
    if val_ratio > 0 and len(train_data) > 0 and val_size == 0:
        val_size = 1

    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    return train_data, val_data, test_data
    raise NotImplementedError("make_sentiment_dataset을 구현하세요.")


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        sample = self.data[idx]
        text = sample["text"]
        label = int(sample["label"])

        try:
            ids = self.tokenizer.encode(text, add_bos_eos=True)
        except TypeError:
            ids = self.tokenizer.encode(text)

        ids = ids[: self.max_length]

        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)
        return input_ids, label
        raise NotImplementedError("ReviewSentimentDataset.__getitem__을 구현하세요.")


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)    
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)
        return
        raise NotImplementedError("GPTForSequenceClassification.__init__을 구현하세요.")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)

        x = self.gpt.final_norm(x)
        last_hidden = x[:, -1, :]
        logits = self.classifier(self.dropout(last_hidden))

        if labels is None:
            return logits

        loss = torch.nn.functional.cross_entropy(logits, labels.long())
        return loss, logits
        raise NotImplementedError("GPTForSequenceClassification.forward를 구현하세요.")


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    if len(train_loader) == 0:
        return float("nan"), 0.0

    model.to(device)
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy
    raise NotImplementedError("train_epoch_sentiment를 구현하세요.")


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    if len(data_loader) == 0:
        return float("nan"), 0.0
    
    model.to(device)
    was_training = model.training
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device, dtype=torch.long)

            loss, logits = model(input_ids, labels=labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    if was_training:
        model.train()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy
    raise NotImplementedError("evaluate_sentiment를 구현하세요.")
