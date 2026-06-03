# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def save_token_ids(token_ids: list[int], path: str | Path) -> None:
    """토큰화된 ID 리스트를 재사용할 수 있도록 Tensor 파일로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(token_ids, dtype=torch.long), path)


def load_token_ids(path: str | Path) -> list[int]:
    """save_token_ids()로 저장한 token ID 리스트를 읽습니다."""
    cached = torch.load(Path(path), map_location="cpu")
    if isinstance(cached, torch.Tensor):
        return cached.long().tolist()
    return list(cached)


def get_or_create_token_ids(
    corpus: str,
    tokenizer,
    cache_path: str | Path,
    add_bos_eos: bool = False,
    force_retokenize: bool = False,
) -> list[int]:
    """캐시가 있으면 token_ids를 읽고, 없으면 corpus를 encode한 뒤 저장합니다."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_retokenize:
        return load_token_ids(cache_path)

    try:
        token_ids = tokenizer.encode(corpus, add_bos_eos=add_bos_eos)
    except TypeError:
        token_ids = tokenizer.encode(corpus)
        if add_bos_eos:
            if hasattr(tokenizer, "get_bos_id"):
                token_ids = [tokenizer.get_bos_id()] + token_ids
            if hasattr(tokenizer, "get_eos_id"):
                token_ids = token_ids + [tokenizer.get_eos_id()]

    save_token_ids(token_ids, cache_path)
    return token_ids


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

    if was_training:
        model.train()

    return total_loss / num_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)



def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        if isinstance(logits, tuple):
            logits = logits[1]
        logits = logits[:, -1, :]

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            min_value = top_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, -torch.inf),
                logits,
            )

        if temperature <= 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
        if eos_id is not None and torch.all(idx_next == eos_id):
            break
    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    was_training = model.training
    model.eval()
    encoded = tokenizer.encode(start_context)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None
    token_ids = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(decoded)
    if was_training:
        model.train()


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    model.to(device)
    train_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if eval_freq and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(f"step {global_step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                generate_and_print_sample(
                    model,
                    tokenizer,
                    device,
                    start_context,
                    context_size=model.config.get("context_length", 256),
                )
                model.train()

            if ckpt_freq and global_step % ckpt_freq == 0:
                save_checkpoint(model, optimizer, epoch, global_step, f"checkpoint_step_{global_step}.pt")

        train_losses.append(epoch_loss / max(1, num_batches))

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
