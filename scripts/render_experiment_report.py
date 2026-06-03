# -*- coding: utf-8 -*-
"""실험 결과를 Markdown, HTML, PNG 그래프로 렌더링합니다."""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
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


def fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def fmt_minutes(seconds) -> str:
    if value_is_missing(seconds):
        return "-"
    return f"{float(seconds) / 60:.1f}분"


def value_is_missing(value) -> bool:
    return value is None or value == ""


def label_for_result(path: Path, data: dict) -> str:
    name = path.stem.lower()
    if data.get("loaded_checkpoint") or "checkpoint" in name:
        return "사전 학습 checkpoint"
    if "random" in name:
        return "랜덤 초기화"
    return path.stem


def read_json_results(outputs_dir: Path) -> list[tuple[Path, dict]]:
    if not outputs_dir.exists():
        return []

    results = []
    for path in sorted(outputs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results.append((path, data))
    return results


def parse_pretrain_log(path: Path) -> dict:
    if not path.exists():
        return {"rows": [], "epoch_train_losses": [], "source": str(path)}

    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    pattern = re.compile(
        r"(?:step|final step) (?P<step>\d+): train loss (?P<train>[0-9.]+), val loss (?P<val>[0-9.]+)"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "step": int(match.group("step")),
                "train_loss": float(match.group("train")),
                "val_loss": float(match.group("val")),
            }
        )

    epoch_train_losses = []
    epoch_match = re.search(r"epoch_train_losses:\s*(\[[^\n]+\])", text)
    if epoch_match:
        try:
            epoch_train_losses = list(ast.literal_eval(epoch_match.group(1)))
        except (ValueError, SyntaxError):
            epoch_train_losses = []

    return {
        "rows": rows,
        "epoch_train_losses": epoch_train_losses,
        "source": str(path),
    }


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pretrain_loss_plot(pretrain: dict, output_dir: Path) -> Path | None:
    rows = pretrain["rows"]
    if not rows:
        return None

    path = output_dir / "pretrain_loss.png"
    steps = [row["step"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_losses, marker="o", linewidth=2, label="Train loss")
    plt.plot(steps, val_losses, marker="o", linewidth=2, label="Validation loss")
    plt.title("Pretraining Loss")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_finetune_plots(results: list[tuple[Path, dict]], output_dir: Path) -> dict[str, Path | None]:
    if not results:
        return {
            "loss": None,
            "accuracy": None,
            "test_metrics": None,
        }

    loss_path = output_dir / "finetune_loss.png"
    acc_path = output_dir / "finetune_accuracy.png"
    test_path = output_dir / "finetune_test_metrics.png"

    plt.figure(figsize=(9, 5))
    has_loss = False
    for path, data in results:
        history = data.get("history", [])
        if not history:
            continue
        label = label_for_result(path, data)
        epochs = [row["epoch"] for row in history]
        val_loss = [row["val_loss"] for row in history]
        train_loss = [row["train_loss"] for row in history]
        plt.plot(epochs, train_loss, linestyle="--", marker="o", label=f"{label} train")
        plt.plot(epochs, val_loss, marker="o", label=f"{label} val")
        has_loss = True
    if has_loss:
        plt.title("Fine-tuning Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    has_acc = False
    for path, data in results:
        history = data.get("history", [])
        if not history:
            continue
        label = label_for_result(path, data)
        epochs = [row["epoch"] for row in history]
        val_acc = [row["val_accuracy"] for row in history]
        train_acc = [row["train_accuracy"] for row in history]
        plt.plot(epochs, train_acc, linestyle="--", marker="o", label=f"{label} train")
        plt.plot(epochs, val_acc, marker="o", label=f"{label} val")
        has_acc = True
    if has_acc:
        plt.title("Fine-tuning Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_path, dpi=160)
    plt.close()

    labels = [label_for_result(path, data) for path, data in results]
    test_acc = [data.get("test_accuracy") for _, data in results]
    test_loss = [data.get("test_loss") for _, data in results]
    if any(v is not None for v in test_acc):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, test_acc, color="#2563eb")
        axes[0].set_title("Test Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=12)
        axes[1].bar(labels, test_loss, color="#dc2626")
        axes[1].set_title("Test Loss")
        axes[1].tick_params(axis="x", rotation=12)
        fig.tight_layout()
        fig.savefig(test_path, dpi=160)
        plt.close(fig)
    else:
        test_path = None

    return {
        "loss": loss_path if has_loss else None,
        "accuracy": acc_path if has_acc else None,
        "test_metrics": test_path,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_summary_md(
    results: list[tuple[Path, dict]],
    pretrain: dict,
    image_paths: dict[str, Path | None],
    output_dir: Path,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    checkpoint_done = any(data.get("loaded_checkpoint") for _, data in results)
    random_done = any(not data.get("loaded_checkpoint") and "random" in path.stem for path, data in results)

    lines = [
        "# mini GPT 실험 결과 요약",
        "",
        f"생성 시각: `{now}`",
        "",
        "## 1. 진행 상태",
        "",
        markdown_table(
            ["항목", "상태"],
            [
                ["사전 학습 결과", "있음" if pretrain["rows"] else "없음"],
                ["Checkpoint 미세 조정", "있음" if checkpoint_done else "없음"],
                ["랜덤 초기화 미세 조정", "있음" if random_done else "아직 없음"],
                ["가설 A 비교 가능 여부", "가능" if checkpoint_done and random_done else "랜덤 초기화 결과 필요"],
            ],
        ),
        "",
    ]

    if pretrain["rows"]:
        last = pretrain["rows"][-1]
        lines.extend(
            [
                "## 2. 사전 학습 결과",
                "",
                markdown_table(
                    ["마지막 step", "Train loss", "Validation loss"],
                    [[str(last["step"]), fmt_float(last["train_loss"]), fmt_float(last["val_loss"])]],
                ),
                "",
            ]
        )
        if pretrain["epoch_train_losses"]:
            epoch_rows = [
                [str(i + 1), fmt_float(loss)]
                for i, loss in enumerate(pretrain["epoch_train_losses"])
            ]
            lines.extend(
                [
                    "### Epoch별 train loss",
                    "",
                    markdown_table(["Epoch", "Train loss"], epoch_rows),
                    "",
                ]
            )
        if image_paths.get("pretrain"):
            lines.extend(["![Pretraining loss](pretrain_loss.png)", ""])

    if results:
        rows = []
        for path, data in results:
            history = data.get("history", [])
            last = history[-1] if history else {}
            rows.append(
                [
                    label_for_result(path, data),
                    path.name,
                    fmt_pct(last.get("val_accuracy")),
                    fmt_float(last.get("val_loss")),
                    fmt_pct(data.get("test_accuracy")),
                    fmt_float(data.get("test_loss")),
                    fmt_minutes(data.get("elapsed_seconds")),
                ]
            )
        lines.extend(
            [
                "## 3. 감성 분류 미세 조정 결과",
                "",
                markdown_table(
                    ["실험", "파일", "최종 Val Acc", "최종 Val Loss", "Test Acc", "Test Loss", "시간"],
                    rows,
                ),
                "",
            ]
        )
        for key, title in [
            ("accuracy", "Fine-tuning accuracy"),
            ("loss", "Fine-tuning loss"),
            ("test_metrics", "Test metrics"),
        ]:
            image = image_paths.get(key)
            if image:
                lines.extend([f"![{title}]({image.name})", ""])

    lines.extend(
        [
            "## 4. 다음에 할 일",
            "",
            "- `outputs/finetune_random_results.json`이 아직 없다면 랜덤 초기화 실험을 실행한다.",
            "- 랜덤 초기화 결과가 생기면 `python scripts/render_experiment_report.py`를 다시 실행한다.",
            "- 발표에는 `finetune_test_metrics.png`, `finetune_accuracy.png`, `pretrain_loss.png`를 캡처하거나 삽입한다.",
            "",
        ]
    )

    return "\n".join(lines)


def html_img(path: Path | None, alt: str) -> str:
    if path is None:
        return '<div class="empty">아직 그래프 데이터가 없습니다.</div>'
    return f'<img src="{html.escape(path.name)}" alt="{html.escape(alt)}">'


def build_dashboard_html(
    results: list[tuple[Path, dict]],
    pretrain: dict,
    image_paths: dict[str, Path | None],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_done = any(data.get("loaded_checkpoint") for _, data in results)
    random_done = any(not data.get("loaded_checkpoint") and "random" in path.stem for path, data in results)

    best_test = None
    if results:
        best = max(results, key=lambda item: item[1].get("test_accuracy", -1))
        best_test = (label_for_result(best[0], best[1]), best[1].get("test_accuracy"))

    rows = []
    for path, data in results:
        history = data.get("history", [])
        last = history[-1] if history else {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(label_for_result(path, data))}</td>"
            f"<td><code>{html.escape(path.name)}</code></td>"
            f"<td>{fmt_pct(last.get('val_accuracy'))}</td>"
            f"<td>{fmt_float(last.get('val_loss'))}</td>"
            f"<td>{fmt_pct(data.get('test_accuracy'))}</td>"
            f"<td>{fmt_float(data.get('test_loss'))}</td>"
            f"<td>{fmt_minutes(data.get('elapsed_seconds'))}</td>"
            "</tr>"
        )

    pretrain_last = pretrain["rows"][-1] if pretrain["rows"] else None
    warning = ""
    if checkpoint_done and not random_done:
        warning = """
        <div class="warn">
          가설 A 비교를 완성하려면 랜덤 초기화 실험이 필요합니다.
          <code>python scripts/finetune_local.py --tokenizer-path data/tokenizer_vocab3000.json --results-path outputs/finetune_random_results.json --save-model-path checkpoints/finetune_random.pt</code>
        </div>
        """

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mini GPT 실험 대시보드</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e0ea;
      --accent: #1864ab;
      --warn-bg: #fff7dd;
      --warn-ink: #7a4b00;
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
      color: #fff;
      background: #172033;
    }}
    header .inner, main {{
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 4vw, 48px); }}
    header p {{ margin: 0; color: #d6dde8; }}
    main {{ padding: 24px 0 60px; }}
    section {{
      margin: 18px 0;
      padding: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    h2 {{ margin: 0 0 16px; font-size: 24px; }}
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
    .warn {{
      margin: 14px 0;
      padding: 14px;
      color: var(--warn-ink);
      background: var(--warn-bg);
      border: 1px solid #f5cf70;
      border-radius: 8px;
    }}
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
      <h1>mini GPT 실험 대시보드</h1>
      <p>생성 시각: {html.escape(now)}</p>
    </div>
  </header>
  <main>
    <section>
      <h2>한눈에 보기</h2>
      <div class="metrics">
        <article class="metric"><span>최고 Test Accuracy</span><strong>{fmt_pct(best_test[1]) if best_test else "-"}</strong></article>
        <article class="metric"><span>최고 실험</span><strong>{html.escape(best_test[0]) if best_test else "-"}</strong></article>
        <article class="metric"><span>가설 A 비교</span><strong>{"가능" if checkpoint_done and random_done else "미완성"}</strong></article>
        <article class="metric"><span>사전 학습 마지막 Val Loss</span><strong>{fmt_float(pretrain_last["val_loss"]) if pretrain_last else "-"}</strong></article>
      </div>
      {warning}
    </section>

    <section>
      <h2>감성 분류 미세 조정 결과</h2>
      <table>
        <thead>
          <tr><th>실험</th><th>파일</th><th>최종 Val Acc</th><th>최종 Val Loss</th><th>Test Acc</th><th>Test Loss</th><th>시간</th></tr>
        </thead>
        <tbody>{"".join(rows) if rows else '<tr><td colspan="7">아직 미세 조정 결과가 없습니다.</td></tr>'}</tbody>
      </table>
    </section>

    <section>
      <h2>그래프</h2>
      <div class="grid">
        <article class="chart"><h3>Fine-tuning Accuracy</h3>{html_img(image_paths.get("accuracy"), "Fine-tuning accuracy")}</article>
        <article class="chart"><h3>Fine-tuning Loss</h3>{html_img(image_paths.get("loss"), "Fine-tuning loss")}</article>
        <article class="chart"><h3>Test Metrics</h3>{html_img(image_paths.get("test_metrics"), "Test metrics")}</article>
        <article class="chart"><h3>Pretraining Loss</h3>{html_img(image_paths.get("pretrain"), "Pretraining loss")}</article>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render experiment report assets")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--pretrain-log", default="pretrain_context128_vocab3000.log")
    args = parser.parse_args()
    configure_matplotlib_font()

    output_dir = ROOT / args.outputs_dir
    ensure_output_dir(output_dir)

    results = read_json_results(output_dir)
    pretrain = parse_pretrain_log(ROOT / args.pretrain_log)

    image_paths = save_finetune_plots(results, output_dir)
    image_paths["pretrain"] = save_pretrain_loss_plot(pretrain, output_dir)

    summary_md = build_summary_md(results, pretrain, image_paths, output_dir)
    summary_path = output_dir / "experiment_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    dashboard_html = build_dashboard_html(results, pretrain, image_paths)
    dashboard_path = output_dir / "experiment_dashboard.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")

    print(f"Markdown 요약 저장: {summary_path}")
    print(f"HTML 대시보드 저장: {dashboard_path}")
    for key, path in image_paths.items():
        if path is not None:
            print(f"{key} 그래프 저장: {path}")


if __name__ == "__main__":
    main()
