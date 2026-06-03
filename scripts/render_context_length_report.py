# -*- coding: utf-8 -*-
"""context_length 64/128 사전 학습 실험 결과를 시각화합니다."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent.parent


def configure_matplotlib_font() -> None:
    preferred_fonts = ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def fmt_float(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_minutes(seconds) -> str:
    if seconds is None:
        return "-"
    return f"{float(seconds) / 60:.1f}분"


def read_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_len(num_tokens: int, context_length: int, stride: int) -> int:
    limit = num_tokens - context_length
    if limit <= 0:
        return 0
    return (limit - 1) // stride + 1


def summarize(path: Path, data: dict | None) -> dict:
    if data is None:
        return {
            "path": path,
            "exists": False,
            "label": path.stem,
        }

    args = data.get("args", {})
    config = data.get("config", {})
    history = data.get("history", {})
    context_length = int(config.get("context_length", args.get("context_length", 0)))
    stride = args.get("stride") or context_length
    batch_size = int(args.get("batch_size", 1))
    train_tokens = int(data.get("train_tokens", 0))
    val_tokens = int(data.get("val_tokens", 0))
    train_samples = dataset_len(train_tokens, context_length, int(stride))
    val_samples = dataset_len(val_tokens, context_length, int(stride))
    train_batches = train_samples // batch_size
    val_batches = math.ceil(val_samples / batch_size) if batch_size else 0
    train_eval_losses = history.get("train_eval_losses", [])
    val_losses = history.get("val_losses", [])

    return {
        "path": path,
        "exists": True,
        "label": f"context_length {context_length}",
        "context_length": context_length,
        "stride": stride,
        "batch_size": batch_size,
        "epochs": args.get("epochs"),
        "eval_steps": history.get("eval_steps", []),
        "train_eval_losses": train_eval_losses,
        "val_losses": val_losses,
        "final_train_loss": train_eval_losses[-1] if train_eval_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "elapsed_seconds": data.get("elapsed_seconds"),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "num_parameters": data.get("num_parameters"),
    }


def save_loss_plot(rows: list[dict], output_dir: Path) -> Path | None:
    existing = [row for row in rows if row.get("exists") and row.get("val_losses")]
    if not existing:
        return None

    path = output_dir / "context_length_loss.png"
    plt.figure(figsize=(9, 5))
    for row in existing:
        steps = row["eval_steps"]
        if not steps:
            steps = list(range(1, len(row["val_losses"]) + 1))
        plt.plot(
            steps,
            row["val_losses"],
            marker="o",
            linewidth=2,
            label=f"{row['label']} val",
        )
        if row["train_eval_losses"]:
            plt.plot(
                steps,
                row["train_eval_losses"],
                linestyle="--",
                linewidth=1.6,
                label=f"{row['label']} train",
            )
    plt.title("Context Length별 Pretraining Loss")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_final_metric_plot(rows: list[dict], output_dir: Path) -> Path | None:
    existing = [row for row in rows if row.get("exists")]
    if not existing:
        return None

    path = output_dir / "context_length_final_metrics.png"
    labels = [str(row["context_length"]) for row in existing]
    final_val = [row["final_val_loss"] for row in existing]
    best_val = [row["best_val_loss"] for row in existing]
    elapsed = [
        (row["elapsed_seconds"] / 60) if row.get("elapsed_seconds") is not None else 0
        for row in existing
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(existing))
    axes[0].bar([i - 0.18 for i in x], final_val, width=0.36, label="Final val loss", color="#2563eb")
    axes[0].bar([i + 0.18 for i in x], best_val, width=0.36, label="Best val loss", color="#16a34a")
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("context_length")
    axes[0].set_xticks(list(x), labels)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, elapsed, color="#dc2626")
    axes[1].set_title("Elapsed Time")
    axes[1].set_xlabel("context_length")
    axes[1].set_ylabel("minutes")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_data_size_plot(rows: list[dict], output_dir: Path) -> Path | None:
    existing = [row for row in rows if row.get("exists")]
    if not existing:
        return None

    path = output_dir / "context_length_batches.png"
    labels = [str(row["context_length"]) for row in existing]
    train_batches = [row["train_batches"] for row in existing]
    val_batches = [row["val_batches"] for row in existing]

    plt.figure(figsize=(8, 4.6))
    x = range(len(existing))
    plt.bar([i - 0.18 for i in x], train_batches, width=0.36, label="Train batches", color="#475569")
    plt.bar([i + 0.18 for i in x], val_batches, width=0.36, label="Validation batches", color="#94a3b8")
    plt.title("Context Length별 Batch 수")
    plt.xlabel("context_length")
    plt.ylabel("batches per epoch")
    plt.xticks(list(x), labels)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_summary(rows: list[dict], images: dict[str, Path | None]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table_rows = []
    for row in rows:
        if not row.get("exists"):
            table_rows.append([row["path"].name, "없음", "-", "-", "-", "-", "-"])
            continue
        table_rows.append(
            [
                str(row["context_length"]),
                row["path"].name,
                fmt_float(row["final_val_loss"]),
                fmt_float(row["best_val_loss"]),
                fmt_float(row["final_train_loss"]),
                fmt_minutes(row["elapsed_seconds"]),
                str(row["train_batches"]),
            ]
        )

    lines = [
        "# 가설 D context_length 비교 결과",
        "",
        f"생성 시각: `{now}`",
        "",
        "## 결과 표",
        "",
        markdown_table(
            [
                "context_length",
                "결과 파일",
                "Final Val Loss",
                "Best Val Loss",
                "Final Train Loss",
                "시간",
                "Train batches/epoch",
            ],
            table_rows,
        ),
        "",
        "## 그래프",
        "",
    ]
    for key, title in [
        ("loss", "Loss curve"),
        ("final", "Final metrics"),
        ("batches", "Batch count"),
    ]:
        image = images.get(key)
        if image is not None:
            lines.extend([f"![{title}]({image.name})", ""])

    lines.extend(
        [
            "## 해석 기준",
            "",
            "- `context_length 128`의 validation loss가 더 낮으면 긴 문맥이 도움이 됐다고 볼 수 있다.",
            "- `context_length 128`의 시간이 훨씬 길고 loss 차이가 작으면 비용 대비 이득이 작다고 볼 수 있다.",
            "- 두 실험은 tokenizer, vocab_size, emb_dim, n_heads, n_layers, batch_size, epochs를 같게 맞춰야 비교가 깔끔하다.",
            "",
        ]
    )
    return "\n".join(lines)


def img_tag(path: Path | None, alt: str) -> str:
    if path is None:
        return '<div class="empty">아직 그래프 데이터가 없습니다.</div>'
    return f'<img src="{html.escape(path.name)}" alt="{html.escape(alt)}">'


def build_html(rows: list[dict], images: dict[str, Path | None]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing = [row for row in rows if row.get("exists")]
    best = min(existing, key=lambda row: row["best_val_loss"]) if existing else None
    table = []
    for row in rows:
        if not row.get("exists"):
            table.append(
                "<tr>"
                f"<td><code>{html.escape(row['path'].name)}</code></td>"
                "<td colspan='7'>결과 파일 없음</td>"
                "</tr>"
            )
            continue
        table.append(
            "<tr>"
            f"<td>{row['context_length']}</td>"
            f"<td><code>{html.escape(row['path'].name)}</code></td>"
            f"<td>{fmt_float(row['final_val_loss'])}</td>"
            f"<td>{fmt_float(row['best_val_loss'])}</td>"
            f"<td>{fmt_float(row['final_train_loss'])}</td>"
            f"<td>{fmt_minutes(row['elapsed_seconds'])}</td>"
            f"<td>{row['train_batches']}</td>"
            f"<td>{row['num_parameters']:,}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>가설 D context_length 비교</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e0ea;
      --accent: #155eef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.6;
    }}
    header {{
      padding: 40px 24px;
      color: white;
      background: #172033;
    }}
    header .inner, main {{
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 4vw, 46px); }}
    header p {{ margin: 0; color: #d6dde8; }}
    main {{ padding: 24px 0 60px; }}
    section {{
      margin: 18px 0;
      padding: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    h2 {{ margin: 0 0 16px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 16px;
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 14px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 28px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border: 1px solid var(--line); text-align: left; }}
    th {{ background: #eef3f9; }}
    code {{
      padding: 2px 5px;
      background: #edf1f7;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
    }}
    .chart {{
      padding: 14px;
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .chart img {{ display: block; width: 100%; height: auto; }}
    .empty {{
      padding: 18px;
      color: var(--muted);
      background: #fbfcfe;
      border: 1px dashed #b8c3d1;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <header>
    <div class="inner">
      <h1>가설 D context_length 비교</h1>
      <p>생성 시각: {html.escape(now)}</p>
    </div>
  </header>
  <main>
    <section>
      <h2>한눈에 보기</h2>
      <div class="metrics">
        <article class="metric"><span>Best context_length</span><strong>{best['context_length'] if best else '-'}</strong></article>
        <article class="metric"><span>Best Val Loss</span><strong>{fmt_float(best['best_val_loss']) if best else '-'}</strong></article>
        <article class="metric"><span>비교 가능 여부</span><strong>{"가능" if len(existing) >= 2 else "결과 부족"}</strong></article>
        <article class="metric"><span>결과 파일 수</span><strong>{len(existing)}</strong></article>
      </div>
    </section>

    <section>
      <h2>결과 표</h2>
      <table>
        <thead>
          <tr><th>context</th><th>파일</th><th>Final Val Loss</th><th>Best Val Loss</th><th>Final Train Loss</th><th>시간</th><th>Train batches/epoch</th><th>Params</th></tr>
        </thead>
        <tbody>{''.join(table)}</tbody>
      </table>
    </section>

    <section>
      <h2>그래프</h2>
      <div class="grid">
        <article class="chart"><h3>Loss Curve</h3>{img_tag(images.get('loss'), 'loss curve')}</article>
        <article class="chart"><h3>Final Metrics</h3>{img_tag(images.get('final'), 'final metrics')}</article>
        <article class="chart"><h3>Batch Count</h3>{img_tag(images.get('batches'), 'batch count')}</article>
      </div>
    </section>

    <section>
      <h2>해석 기준</h2>
      <p><code>context_length 128</code>의 validation loss가 더 낮으면 긴 문맥을 보는 효과가 있었다고 볼 수 있습니다.</p>
      <p>반대로 loss 차이가 거의 없고 시간이 더 길면, 이 데이터와 모델 크기에서는 128의 비용 대비 이득이 작다고 해석할 수 있습니다.</p>
    </section>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render context_length comparison report")
    parser.add_argument("--context64", default="outputs/pretrain_context64_results.json")
    parser.add_argument("--context128", default="outputs/pretrain_context128_results.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--html-path", default="outputs/context_length_dashboard.html")
    parser.add_argument("--md-path", default="outputs/context_length_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_font()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [ROOT / args.context64, ROOT / args.context128]
    rows = [summarize(path, read_result(path)) for path in paths]
    rows.sort(key=lambda row: row.get("context_length", 10**9))

    images = {
        "loss": save_loss_plot(rows, output_dir),
        "final": save_final_metric_plot(rows, output_dir),
        "batches": save_data_size_plot(rows, output_dir),
    }

    md_path = ROOT / args.md_path
    html_path = ROOT / args.html_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_summary(rows, images), encoding="utf-8")
    html_path.write_text(build_html(rows, images), encoding="utf-8")

    print(f"Markdown 저장: {md_path}")
    print(f"HTML 저장: {html_path}")
    for path in images.values():
        if path is not None:
            print(f"그래프 저장: {path}")


if __name__ == "__main__":
    main()
