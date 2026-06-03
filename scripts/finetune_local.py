# -*- coding: utf-8 -*-
"""로컬에서 NSMC 감성 분류 미세 조정 실험을 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bpe import BPETokenizer
from finetune import GPTForSequenceClassification
from model import GPTModel
from train import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini GPT sentiment fine-tuning")
    parser.add_argument("--train-data", default="data/nsmc_sentiment_train.jsonl")
    parser.add_argument("--val-data", default="data/nsmc_sentiment_val.jsonl")
    parser.add_argument("--test-data", default="data/nsmc_sentiment_test.jsonl")
    parser.add_argument("--tokenizer-path", default="data/tokenizer_vocab3000.json")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--results-path", default="outputs/finetune_results.json")
    parser.add_argument("--save-model-path", default="checkpoints/finetune_final.pt")
    parser.add_argument(
        "--sentiment-cache-dir",
        default="data/sentiment_cache",
        help="감성 분류 리뷰를 미리 token ID tensor로 저장할 cache 디렉터리.",
    )
    parser.add_argument(
        "--force-sentiment-retokenize",
        action="store_true",
        help="감성 분류 token cache가 있어도 다시 생성합니다.",
    )

    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--trainable",
        choices=["all", "classifier", "last-block"],
        default="all",
        help="미세 조정할 파라미터 범위.",
    )
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-val-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def read_jsonl(path: str | Path, max_examples: int = 0) -> list[dict]:
    data_path = ROOT / path
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} 파일이 없습니다. 먼저 `python download_data.py`를 실행하세요.")

    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_examples > 0 and len(rows) >= max_examples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(path: str | Path, vocab_size: int) -> BPETokenizer:
    tokenizer_path = ROOT / path
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.load(tokenizer_path)
    actual_vocab_size = len(tokenizer.id_to_token)
    if actual_vocab_size != vocab_size:
        raise ValueError(
            f"tokenizer vocab mismatch: 명령어 vocab_size={vocab_size}, "
            f"파일 vocab_size={actual_vocab_size}."
        )
    return tokenizer


def sentiment_cache_path(args: argparse.Namespace, split: str, max_examples: int) -> Path:
    tokenizer_name = Path(args.tokenizer_path).stem
    limit = max_examples if max_examples > 0 else "all"
    filename = (
        f"{split}_{tokenizer_name}_vocab{args.vocab_size}_"
        f"ctx{args.context_length}_max{limit}.pt"
    )
    return ROOT / args.sentiment_cache_dir / filename


def expected_cache_meta(
    args: argparse.Namespace,
    split: str,
    data_path: str | Path,
    max_examples: int,
    num_examples: int,
    pad_id: int,
) -> dict:
    resolved_data_path = ROOT / data_path
    stat = resolved_data_path.stat()
    return {
        "split": split,
        "data_path": str(resolved_data_path.resolve()),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "max_examples": max_examples,
        "num_examples": num_examples,
        "tokenizer_path": str((ROOT / args.tokenizer_path).resolve()),
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "pad_id": pad_id,
        "add_bos_eos": True,
    }


def encode_sentiment_rows(
    data: list[dict],
    tokenizer: BPETokenizer,
    max_length: int,
    pad_id: int,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_rows = []
    labels = []
    for idx, item in enumerate(data, start=1):
        ids = tokenizer.encode(item["text"], add_bos_eos=True)
        ids = ids[:max_length]
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
        input_rows.append(ids)
        labels.append(int(item["label"]))

        if idx % 10_000 == 0:
            print(f"{split} token cache 생성 중: {idx:,}/{len(data):,}", flush=True)

    if input_rows:
        input_ids = torch.tensor(input_rows, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        input_ids = torch.empty((0, max_length), dtype=torch.long)
        label_tensor = torch.empty((0,), dtype=torch.long)
    return input_ids, label_tensor


def load_or_create_sentiment_dataset(
    data: list[dict],
    tokenizer: BPETokenizer,
    args: argparse.Namespace,
    split: str,
    data_path: str | Path,
    max_examples: int,
    pad_id: int,
) -> tuple[TensorDataset, Path]:
    cache_path = sentiment_cache_path(args, split, max_examples)
    expected_meta = expected_cache_meta(args, split, data_path, max_examples, len(data), pad_id)

    if cache_path.exists() and not args.force_sentiment_retokenize:
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("meta") == expected_meta:
            print(f"{split} token cache 로드: {cache_path}", flush=True)
            return TensorDataset(cached["input_ids"], cached["labels"]), cache_path
        print(f"{split} token cache 설정 변경 감지, 다시 생성: {cache_path}", flush=True)

    print(f"{split} token cache 생성 시작: {cache_path}", flush=True)
    input_ids, labels = encode_sentiment_rows(
        data,
        tokenizer,
        max_length=args.context_length,
        pad_id=pad_id,
        split=split,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": input_ids,
            "labels": labels,
            "meta": expected_meta,
        },
        cache_path,
    )
    print(f"{split} token cache 저장: {cache_path}", flush=True)
    return TensorDataset(input_ids, labels), cache_path


def set_trainable(model: GPTForSequenceClassification, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = mode == "all"

    if mode in {"classifier", "last-block"}:
        for param in model.classifier.parameters():
            param.requires_grad = True

    if mode == "last-block":
        for param in model.gpt.final_norm.parameters():
            param.requires_grad = True
        for param in model.gpt.blocks[-1].parameters():
            param.requires_grad = True


def run_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += batch_size

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device).long()

            loss, logits = model(input_ids, labels=labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += batch_size

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def main() -> None:
    args = parse_args()
    start_time = time.time()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer_path, args.vocab_size)
    pad_id = tokenizer.get_pad_id()

    train_data = read_jsonl(args.train_data, args.max_train_examples)
    val_data = read_jsonl(args.val_data, args.max_val_examples)
    test_data = read_jsonl(args.test_data, args.max_test_examples)

    train_dataset, train_cache_path = load_or_create_sentiment_dataset(
        train_data,
        tokenizer,
        args,
        split="train",
        data_path=args.train_data,
        max_examples=args.max_train_examples,
        pad_id=pad_id,
    )
    val_dataset, val_cache_path = load_or_create_sentiment_dataset(
        val_data,
        tokenizer,
        args,
        split="val",
        data_path=args.val_data,
        max_examples=args.max_val_examples,
        pad_id=pad_id,
    )
    test_dataset, test_cache_path = load_or_create_sentiment_dataset(
        test_data,
        tokenizer,
        args,
        split="test",
        data_path=args.test_data,
        max_examples=args.max_test_examples,
        pad_id=pad_id,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }

    device = select_device()
    backbone = GPTModel(config).to(device)

    loaded_checkpoint = None
    if args.checkpoint_path is not None:
        loaded_checkpoint = str((ROOT / args.checkpoint_path).resolve())
        epoch, global_step = load_checkpoint(backbone, optimizer=None, path=ROOT / args.checkpoint_path, device=device)
        print(f"checkpoint 로드: epoch={epoch}, global_step={global_step}", flush=True)

    model = GPTForSequenceClassification(backbone, num_labels=2, drop_rate=args.drop_rate, pad_id=pad_id).to(device)
    set_trainable(model, args.trainable)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("학습할 파라미터가 없습니다. --trainable 설정을 확인하세요.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"device: {device}", flush=True)
    print(f"train/val/test examples: {len(train_data)}/{len(val_data)}/{len(test_data)}", flush=True)
    print(f"trainable: {args.trainable}", flush=True)
    print(f"checkpoint: {loaded_checkpoint or 'random initialization'}", flush=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch}: train loss {train_loss:.4f}, train acc {train_acc:.4f}, "
            f"val loss {val_loss:.4f}, val acc {val_acc:.4f}",
            flush=True,
        )

    test_loss, test_acc = evaluate(model, test_loader, device)
    elapsed_seconds = time.time() - start_time

    save_model_path = ROOT / args.save_model_path
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "args": vars(args),
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        },
        save_model_path,
    )

    results = {
        "args": vars(args),
        "config": config,
        "device": str(device),
        "loaded_checkpoint": loaded_checkpoint,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
        "sentiment_cache_paths": {
            "train": str(train_cache_path.resolve()),
            "val": str(val_cache_path.resolve()),
            "test": str(test_cache_path.resolve()),
        },
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "saved_model_path": str(save_model_path.resolve()),
        "elapsed_seconds": elapsed_seconds,
    }
    results_path = ROOT / args.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"test loss {test_loss:.4f}, test acc {test_acc:.4f}", flush=True)
    print(f"model 저장: {save_model_path}", flush=True)
    print(f"실험 결과 저장: {results_path}", flush=True)
    print(f"소요 시간: {elapsed_seconds / 60:.2f}분", flush=True)


if __name__ == "__main__":
    main()
