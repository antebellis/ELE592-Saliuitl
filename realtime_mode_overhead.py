import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np


METRIC_RE = re.compile(r"^(Unsuccesful Attacks|Detected Attacks|Successful Attacks):\s*([0-9eE+.\-]+)\s*$")


def load_npy_len(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    arr = np.load(path, allow_pickle=True)
    return int(len(arr))


def parse_metrics(stdout: str) -> dict:
    metrics = {}
    for line in stdout.splitlines():
        m = METRIC_RE.match(line.strip())
        if m:
            key = m.group(1)
            val = float(m.group(2))
            if key == "Unsuccesful Attacks":
                metrics["unsuccessful_attacks_rate"] = val
            elif key == "Detected Attacks":
                metrics["detected_attacks_rate"] = val
            elif key == "Successful Attacks":
                metrics["successful_attacks_rate"] = val
    return metrics


def run_mode(args, mode_name: str, mode_flags: list[str], default_frames: int | None = None) -> dict:
    savedir = Path(args.out_root) / mode_name
    savedir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "saliuitl.py",
        "--dataset",
        args.dataset,
        "--imgdir",
        args.imgdir,
        "--patch_imgdir",
        args.patch_imgdir,
        "--det_net_path",
        args.det_net_path,
        "--det_net",
        args.det_net,
        "--effective_files",
        args.effective_files,
        "--n_patches",
        args.n_patches,
        "--inpaint",
        args.inpaint,
        "--det_mode",
        args.det_mode,
        "--ensemble_step",
        str(args.ensemble_step),
        "--inpainting_step",
        str(args.inpainting_step),
        "--lim",
        str(args.lim),
        "--save_scores",
        "--performance",
        "--performance_det",
        "--savedir",
        str(savedir),
    ] + mode_flags

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, text=True, capture_output=True)
    elapsed_sec = time.perf_counter() - t0

    log_path = savedir / "run.log"
    log_path.write_text((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Mode '{mode_name}' failed. See {log_path}")

    base = f"{savedir}_{args.dataset}_{args.det_net}_npatches_{args.n_patches}"
    scores_path = Path(base + f"_ens_{args.ensemble_step}_scores.npy")
    inferred = False
    if scores_path.exists():
        frames = load_npy_len(scores_path)
        if frames == 0 and default_frames is not None:
            frames = int(default_frames)
            inferred = True
    elif default_frames is not None:
        frames = int(default_frames)
        inferred = True
    else:
        raise RuntimeError(f"Could not infer frame count for mode '{mode_name}'.")
    metrics = parse_metrics(proc.stdout)

    fps = float(frames) / max(elapsed_sec, 1e-9)
    rt_factor_vs_10fps = fps / 10.0

    return {
        "frames_inferred_from_reference": inferred,
        "mode": mode_name,
        "command": " ".join(cmd),
        "log_path": str(log_path),
        "frames_processed": frames,
        "elapsed_sec": elapsed_sec,
        "fps": fps,
        "rt_factor_vs_10fps": rt_factor_vs_10fps,
        "meets_10fps": fps >= 10.0,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure stream overhead across Saliuitl operation modes.")
    parser.add_argument("--python", default=".venv312/Scripts/python")
    parser.add_argument("--dataset", default="voc")
    parser.add_argument("--imgdir", default="data/apricot_saliuitl/clean")
    parser.add_argument("--patch_imgdir", default="data/apricot_saliuitl/1p")
    parser.add_argument("--det_net_path", default="checkpoints/final_detection/2dcnn_raw_VOC_5_atk_det.pth")
    parser.add_argument("--det_net", default="2dcnn_raw")
    parser.add_argument("--effective_files", default="effective_1p.npy")
    parser.add_argument("--n_patches", default="1")
    parser.add_argument("--inpaint", default="biharmonic")
    parser.add_argument("--det_mode", default="balanced")
    parser.add_argument("--ensemble_step", type=int, default=5)
    parser.add_argument("--inpainting_step", type=int, default=5)
    parser.add_argument("--lim", type=int, default=1000000)
    parser.add_argument("--out_root", default="realtime_mode_overhead")
    args = parser.parse_args()

    modes = [
        ("detection_only", ["--bypass"]),
        ("detection_plus_recovery", []),
        ("classification_only", ["--bypass_det", "--bypass"]),
        ("always_recovery", ["--bypass_det"]),
    ]

    results = []
    reference_frames = None
    for mode_name, mode_flags in modes:
        result = run_mode(args, mode_name, mode_flags, default_frames=reference_frames)
        if reference_frames is None:
            reference_frames = result["frames_processed"]
        results.append(result)

    by_mode = {r["mode"]: r for r in results}
    full_fps = by_mode["detection_plus_recovery"]["fps"]
    for r in results:
        r["speedup_vs_detection_plus_recovery"] = r["fps"] / max(full_fps, 1e-9)

    summary = {
        "target_stream_fps": 10.0,
        "config": vars(args),
        "results": results,
    }
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
