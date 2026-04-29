import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Prior APRICOT evaluation values from this project runs (stock Saliuitl branch/config).
EVAL = {
    "attacked": {
        "unsuccessful": 0.7787456445993032,
        "detected": 0.3205574912891986,
        "successful": 0.22125435540069685,
        "runtime_min": 11.3,
    },
    "clean": {
        "unsuccessful": 0.9947735191637631,
        "detected": 0.34843205574912894,
        "successful": 0.005226480836236934,
        "runtime_min": 10.4,
    },
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_attack_outcomes(outdir: Path):
    labels = ["Unsuccessful Attack\n(Recovered/Resistant)", "Successful Attack"]
    attacked = [EVAL["attacked"]["unsuccessful"], EVAL["attacked"]["successful"]]
    clean = [EVAL["clean"]["unsuccessful"], EVAL["clean"]["successful"]]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12, 7), dpi=220)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    b1 = ax.bar(x - width / 2, attacked, width, label="APRICOT Attacked", color="#F97316")
    b2 = ax.bar(x + width / 2, clean, width, label="APRICOT Clean Baseline", color="#22C55E")

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", color="white", fontsize=11)

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Rate", color="white", fontsize=13)
    ax.set_title("Saliuitl on APRICOT: Attack Outcome Rates", color="white", fontsize=22, fontweight="bold", pad=14)
    ax.set_xticks(x, labels, color="white")
    ax.tick_params(axis="y", colors="white")
    ax.grid(axis="y", color="#334155", alpha=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(facecolor="#0F172A", edgecolor="#334155", fontsize=12)
    for t in leg.get_texts():
        t.set_color("white")

    fig.tight_layout()
    fig.savefig(outdir / "apricot_eval_outcomes.png")
    plt.close(fig)


def save_detection_and_runtime(outdir: Path):
    labels = ["Attacked", "Clean"]
    detect = [EVAL["attacked"]["detected"], EVAL["clean"]["detected"]]
    runtime = [EVAL["attacked"]["runtime_min"], EVAL["clean"]["runtime_min"]]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(11, 7), dpi=220)
    fig.patch.set_facecolor("#FFF8E8")
    ax1.set_facecolor("#FFF8E8")

    b1 = ax1.bar(x, detect, width=0.5, color=["#FF7F11", "#2A9D8F"], label="Detected Attacks Rate")
    ax1.set_ylim(0, 0.5)
    ax1.set_ylabel("Detection Rate", fontsize=13, color="#1F2937")
    ax1.set_xticks(x, labels, fontsize=12)
    ax1.grid(axis="y", color="#E5D2A4", alpha=0.8)
    for s in ax1.spines.values():
        s.set_visible(False)

    for b in b1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=11, color="#1F2937")

    ax2 = ax1.twinx()
    ax2.plot(x, runtime, color="#1D3557", marker="o", linewidth=2.5, markersize=8, label="Eval Runtime (min)")
    ax2.set_ylim(0, max(runtime) * 1.35)
    ax2.set_ylabel("Runtime (minutes)", fontsize=13, color="#1D3557")
    ax2.tick_params(axis="y", colors="#1D3557")
    for xi, yi in zip(x, runtime):
        ax2.text(xi, yi + 0.35, f"{yi:.1f} min", ha="center", fontsize=11, color="#1D3557")

    ax1.set_title("APRICOT Detection Rate and Evaluation Runtime", fontsize=21, fontweight="bold", color="#1F2937", pad=14)

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper right", frameon=False, fontsize=11)

    fig.tight_layout()
    fig.savefig(outdir / "apricot_eval_detection_runtime.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate slide visuals for APRICOT non-realtime evaluation.")
    parser.add_argument("--outdir", default="slide_visuals_apricot_eval")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    save_attack_outcomes(outdir)
    save_detection_and_runtime(outdir)
    print(f"Saved non-realtime APRICOT visuals to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
