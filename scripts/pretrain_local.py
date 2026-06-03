# -*- coding: utf-8 -*-
"""로컬에서 mini GPT 사전 학습을 실행하는 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bpe import BPETokenizer
from dataset import create_dataloader
from model import GPTModel
from train import get_or_create_token_ids, train_model


class LimitedLoader:
    """DataLoader 앞쪽 일부 batch만 쓰는 얇은 wrapper."""

    def __init__(self, loader, max_batches: int | None):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for batch_idx, batch in enumerate(self.loader):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            yield batch

    def __len__(self):
        if self.max_batches is None:
            return len(self.loader)
        return min(len(self.loader), self.max_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini GPT local pretraining")
    parser.add_argument("--train-text", default="data/nsmc_lm_train.txt")
    parser.add_argument("--val-text", default="data/nsmc_lm_val.txt")
    parser.add_argument(
        "--train-chars",
        type=int,
        default=0,
        help="학습 텍스트 앞쪽 문자 수. 0 이하이면 전체 학습 텍스트를 사용합니다.",
    )
    parser.add_argument(
        "--val-chars",
        type=int,
        default=0,
        help="검증 텍스트 앞쪽 문자 수. 0 이하이면 전체 검증 텍스트를 사용합니다.",
    )
    parser.add_argument("--tokenizer-path", default="data/tokenizer.json")
    parser.add_argument("--train-token-cache", default="data/train_token_ids.pt")
    parser.add_argument("--val-token-cache", default="data/val_token_ids.pt")
    parser.add_argument("--force-retokenize", action="store_true")
    parser.add_argument(
        "--tokenizer-train-chars",
        type=int,
        default=300_000,
        help="토크나이저 학습에 사용할 앞쪽 문자 수. 0 이하이면 전체 텍스트를 사용합니다.",
    )

    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-iter", type=int, default=5)
    parser.add_argument("--ckpt-freq", type=int, default=None)
    parser.add_argument("--start-context", default="이 영화는")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def read_text(path: str) -> str:
    text_path = ROOT / path
    if not text_path.exists():
        raise FileNotFoundError(
            f"{text_path} 파일이 없습니다. 먼저 `python download_data.py`를 실행하세요."
        )
    return text_path.read_text(encoding="utf-8")


def prepare_tokenizer(args: argparse.Namespace, train_text: str, val_text: str) -> BPETokenizer:
    tokenizer_path = ROOT / args.tokenizer_path
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    if tokenizer_path.exists() and not args.force_retokenize:
        tokenizer.load(tokenizer_path)
        print(f"토크나이저 로드: {tokenizer_path}", flush=True)
        return tokenizer

    tokenizer_corpus = train_text + "\n" + val_text
    if args.tokenizer_train_chars > 0:
        tokenizer_corpus = tokenizer_corpus[: args.tokenizer_train_chars]
    print(
        f"토크나이저 학습 시작: vocab_size={args.vocab_size}, "
        f"chars={len(tokenizer_corpus):,}",
        flush=True,
    )
    tokenizer.train(tokenizer_corpus)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"토크나이저 학습 및 저장: {tokenizer_path}", flush=True)
    return tokenizer


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_text = read_text(args.train_text)
    val_text = read_text(args.val_text)
    if args.train_chars > 0:
        train_text = train_text[: args.train_chars]
        print(f"학습 텍스트 제한: {len(train_text):,} chars", flush=True)
    if args.val_chars > 0:
        val_text = val_text[: args.val_chars]
        print(f"검증 텍스트 제한: {len(val_text):,} chars", flush=True)
    tokenizer = prepare_tokenizer(args, train_text, val_text)

    print(f"train token ID 준비 시작: {ROOT / args.train_token_cache}", flush=True)
    train_ids = get_or_create_token_ids(
        train_text,
        tokenizer,
        ROOT / args.train_token_cache,
        force_retokenize=args.force_retokenize,
    )
    print(f"train token ID 준비 완료: {len(train_ids):,} tokens", flush=True)
    print(f"val token ID 준비 시작: {ROOT / args.val_token_cache}", flush=True)
    val_ids = get_or_create_token_ids(
        val_text,
        tokenizer,
        ROOT / args.val_token_cache,
        force_retokenize=args.force_retokenize,
    )
    print(f"val token ID 준비 완료: {len(val_ids):,} tokens", flush=True)

    stride = args.stride if args.stride is not None else args.context_length
    train_loader = create_dataloader(
        train_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=stride,
        drop_last=True,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=stride,
        drop_last=False,
        shuffle=False,
    )

    train_loader = LimitedLoader(train_loader, args.max_train_batches)
    val_loader = LimitedLoader(val_loader, args.max_val_batches)

    if len(train_loader) == 0:
        raise ValueError("train_loader가 비었습니다. 데이터 길이, context_length, batch_size를 확인하세요.")
    if len(val_loader) == 0:
        raise ValueError("val_loader가 비었습니다. validation 데이터 길이와 context_length를 확인하세요.")

    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}", flush=True)
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}", flush=True)
    print(f"config: {config}", flush=True)

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer,
        ckpt_freq=args.ckpt_freq,
    )

    print("train_losses:", train_losses, flush=True)


if __name__ == "__main__":
    main()
