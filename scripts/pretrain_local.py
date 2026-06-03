# -*- coding: utf-8 -*-
"""로컬에서 mini GPT 사전 학습을 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bpe import BPETokenizer
from dataset import create_dataloader
from model import GPTModel
from train import get_or_create_token_ids, plot_losses, train_model


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
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--final-ckpt-path", default="checkpoints/pretrain_final.pt")
    parser.add_argument("--results-path", default="outputs/pretrain_results.json")
    parser.add_argument("--loss-plot-path", default="outputs/pretrain_loss.png")
    parser.add_argument("--show-plot", action="store_true")
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
        actual_vocab_size = len(tokenizer.id_to_token)
        if actual_vocab_size != args.vocab_size:
            raise ValueError(
                f"tokenizer vocab mismatch: 명령어 vocab_size={args.vocab_size}, "
                f"파일 vocab_size={actual_vocab_size}. "
                "--force-retokenize를 붙이거나 tokenizer/cache 파일명을 분리하세요."
            )
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
    actual_vocab_size = len(tokenizer.id_to_token)
    if actual_vocab_size != args.vocab_size:
        raise ValueError(
            f"tokenizer 학습 결과 vocab_size={actual_vocab_size}입니다. "
            f"요청한 vocab_size={args.vocab_size}와 달라 모델 설정을 만들 수 없습니다."
        )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"토크나이저 학습 및 저장: {tokenizer_path}", flush=True)
    return tokenizer


def cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def expected_cache_meta(
    args: argparse.Namespace,
    tokenizer_path: Path,
    text: str,
    split: str,
) -> dict:
    return {
        "split": split,
        "vocab_size": args.vocab_size,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "text_chars": len(text),
        "add_bos_eos": False,
    }


def load_or_create_token_ids_with_meta(
    text: str,
    tokenizer: BPETokenizer,
    cache_path: Path,
    tokenizer_path: Path,
    args: argparse.Namespace,
    split: str,
) -> list[int]:
    expected_meta = expected_cache_meta(args, tokenizer_path, text, split)
    meta_path = cache_meta_path(cache_path)

    if cache_path.exists() and not args.force_retokenize:
        if not meta_path.exists():
            raise ValueError(
                f"{cache_path}는 있지만 {meta_path}가 없습니다. "
                "오래된 cache일 수 있으니 --force-retokenize로 다시 만드세요."
            )
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (actual_meta.get(key), value)
            for key, value in expected_meta.items()
            if actual_meta.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{cache_path} metadata mismatch: {mismatches}. "
                "--force-retokenize를 붙이거나 cache 파일명을 분리하세요."
            )

    token_ids = get_or_create_token_ids(
        text,
        tokenizer,
        cache_path,
        force_retokenize=args.force_retokenize,
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        **expected_meta,
        "num_tokens": len(token_ids),
        "max_token_id": max(token_ids) if token_ids else None,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return token_ids


def main() -> None:
    args = parse_args()
    start_time = time.time()
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
    tokenizer_path = ROOT / args.tokenizer_path

    print(f"train token ID 준비 시작: {ROOT / args.train_token_cache}", flush=True)
    train_ids = load_or_create_token_ids_with_meta(
        train_text,
        tokenizer,
        ROOT / args.train_token_cache,
        tokenizer_path,
        args,
        "train",
    )
    print(f"train token ID 준비 완료: {len(train_ids):,} tokens", flush=True)
    print(f"val token ID 준비 시작: {ROOT / args.val_token_cache}", flush=True)
    val_ids = load_or_create_token_ids_with_meta(
        val_text,
        tokenizer,
        ROOT / args.val_token_cache,
        tokenizer_path,
        args,
        "val",
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

    history = train_model(
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
        ckpt_dir=ROOT / args.ckpt_dir,
        final_ckpt_path=ROOT / args.final_ckpt_path,
        return_history=True,
    )

    print("epoch_train_losses:", history["epoch_train_losses"], flush=True)
    print("train_eval_losses:", history["train_eval_losses"], flush=True)
    print("val_losses:", history["val_losses"], flush=True)

    plot_losses(
        history["train_eval_losses"],
        history["val_losses"],
        steps=history["eval_steps"],
        save_path=ROOT / args.loss_plot_path,
        show=args.show_plot,
    )
    elapsed_seconds = time.time() - start_time

    results = {
        "args": vars(args),
        "config": config,
        "device": str(device),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "train_text_chars": len(train_text),
        "val_text_chars": len(val_text),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "elapsed_seconds": elapsed_seconds,
        "history": history,
        "loss_plot_path": str((ROOT / args.loss_plot_path).resolve()),
        "results_path": str((ROOT / args.results_path).resolve()),
    }
    results_path = ROOT / args.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"loss plot 저장: {ROOT / args.loss_plot_path}", flush=True)
    print(f"실험 결과 저장: {results_path}", flush=True)
    print(f"소요 시간: {elapsed_seconds / 60:.2f}분", flush=True)
    
if __name__ == "__main__":
    main()
