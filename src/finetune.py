# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import csv
import json
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
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                if not text:
                    continue
                rows.append({"text": text, "label": int(row["label"])})
        return rows

    train_rows = read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    if len(train_rows) <= 1:
        val_size = 0
    else:
        val_size = int(len(train_rows) * val_ratio)
        if val_ratio > 0:
            val_size = max(1, val_size)
        val_size = min(val_size, len(train_rows) - 1)

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def write_jsonl(name: str, data: list[dict]) -> None:
            lines = [json.dumps(item, ensure_ascii=False) for item in data]
            Path(output_path, name).write_text("\n".join(lines), encoding="utf-8")

        write_jsonl("sentiment_train.jsonl", train_data)
        write_jsonl("sentiment_val.jsonl", val_data)
        write_jsonl("sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data
 


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
        """text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        ids = self.tokenizer.encode(item["text"], add_bos_eos=True)

        ids = ids[: self.max_length]
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        label = int(item["label"])
        return input_ids, label


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
        pad_id: int = 0,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = pad_id
        emb_dim = gpt_model.config["emb_dim"]
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.final_norm(x)

        valid_lengths = input_ids.ne(self.pad_id).sum(dim=1).clamp(min=1)
        last_token_idx = valid_lengths - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = x[batch_idx, last_token_idx]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        labels = labels.to(input_ids.device).long()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device).long()

            loss, logits = model(input_ids, labels=labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += batch_size

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count
