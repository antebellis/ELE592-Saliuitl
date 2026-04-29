import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    results = {r["mode"]: r for r in data["results"]}
    order = [
        "classification_only",
        "detection_only",
        "detection_plus_recovery",
        "always_recovery",
    ]
    return [results[m] for m in order]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_runtime_fps(results, outdir: Path):
    labels = [
        "Classification Only",
        "Detection Only",
        "Detection + Recovery\n(Saliuitl)",
        "Always Recovery",
    ]
    fps = [r["fps"] for r in results]
    colors = ["#00A676", "#4D9DE0", "#F26419", "#2D3142"]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    bars = ax.barh(labels, fps, color=colors, edgecolor="#D9D9D9", linewidth=0.5)
    ax.axvline(10, color="#FFD166", linestyle="--", linewidth=2, label="10 FPS Target")

    for b, v in zip(bars, fps):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.2f} FPS", va="center", color="white", fontsize=12)

    ax.set_title("APRICOT Runtime Throughput by Mode", color="white", fontsize=22, pad=16, fontweight="bold")
    ax.set_xlabel("Throughput (FPS)", color="white", fontsize=13)
    ax.tick_params(colors="white", labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#2A2F3A", alpha=0.7)
    ax.legend(facecolor="#0E1117", edgecolor="#2A2F3A", labelcolor="white")
    ax.set_xlim(0, max(max(fps) * 1.25, 11))

    fig.tight_layout()
    fig.savefig(outdir / "apricot_runtime_fps.png", transparent=False)
    plt.close(fig)


def save_overhead_seconds(results, outdir: Path):
    labels = ["Cls Only", "Det Only", "Det+Rec", "Always Rec"]
    spf = [r["elapsed_sec"] / r["frames_processed"] for r in results]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor("#F7F3E9")
    ax.set_facecolor("#F7F3E9")

    bars = ax.bar(x, spf, color=["#7BD389", "#59A5D8", "#E98039", "#3D405B"], width=0.6)
    for b, v in zip(bars, spf):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}s", ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel("Seconds per Frame (lower is better)", fontsize=13)
    ax.set_title("Per-Frame Cost on APRICOT", fontsize=22, pad=16, fontweight="bold")
    ax.grid(axis="y", color="#D9CDBB", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "apricot_seconds_per_frame.png", transparent=False)
    plt.close(fig)


def save_scorecard(results, outdir: Path):
    by_mode = {r["mode"]: r for r in results}
    sal = by_mode["detection_plus_recovery"]
    det = by_mode["detection_only"]
    always = by_mode["always_recovery"]
    cls = by_mode["classification_only"]

    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#101820")

    title = "Saliuitl on APRICOT: Real-Time Tradeoff Snapshot"
    ax.text(0.02, 0.93, title, color="white", fontsize=24, fontweight="bold", transform=ax.transAxes)

    cards = [
        ("Dataset Slice", f"{sal['frames_processed']} effective frames", "#2D6A4F"),
        ("Baseline", f"{cls['fps']:.2f} FPS (classification only)", "#1D3557"),
        ("Saliuitl Full", f"{sal['fps']:.2f} FPS\nSuccess atk rate {sal['successful_attacks_rate']:.3f}", "#BC6C25"),
        ("Always Recovery", f"{always['fps']:.2f} FPS\nSuccess atk rate {always['successful_attacks_rate']:.3f}", "#6D597A"),
        ("Two-Stage Gain", f"{sal['fps']/always['fps']:.2f}x faster than always-recovery", "#D62828"),
        ("Detection Cost", f"Det-only: {det['fps']:.2f} FPS", "#264653"),
    ]

    xs = [0.03, 0.35, 0.67, 0.03, 0.35, 0.67]
    ys = [0.58, 0.58, 0.58, 0.20, 0.20, 0.20]

    for (hdr, body, color), x, y in zip(cards, xs, ys):
        rect = plt.Rectangle((x, y), 0.28, 0.28, transform=ax.transAxes, facecolor=color, alpha=0.9, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.015, y + 0.20, hdr, color="white", fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.015, y + 0.08, body, color="white", fontsize=13, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(outdir / "apricot_scorecard.png", transparent=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create slide-friendly APRICOT visuals from benchmark summary.")
    parser.add_argument(
        "--summary",
        default="realtime_mode_overhead_apricot_full/summary.json",
        help="Path to realtime mode overhead summary JSON.",
    )
    parser.add_argument("--outdir", default="slide_visuals_apricot", help="Output directory for PNG visuals.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    results = load_results(summary_path)
    save_runtime_fps(results, outdir)
    save_overhead_seconds(results, outdir)
    save_scorecard(results, outdir)
    print(f"Saved visuals to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
