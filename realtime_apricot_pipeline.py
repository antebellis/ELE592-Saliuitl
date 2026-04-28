import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def det_threshold_from_args(det_mode: str, nn_det_threshold: float) -> float:
    if det_mode == "manual":
        return nn_det_threshold
    if det_mode == "balanced":
        return 0.50
    if det_mode == "attack_recall":
        return 0.35
    return 0.70


def pct(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return np.load(path)


def run_saliuitl(args, clean_flag: bool) -> dict:
    repo_root = Path(__file__).resolve().parent
    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "saliuitl.py",
        "--det_mode",
        args.det_mode,
        "--nn_det_threshold",
        str(args.nn_det_threshold),
        "--inpaint",
        "biharmonic",
        "--imgdir",
        args.imgdir,
        "--patch_imgdir",
        args.patch_imgdir,
        "--dataset",
        args.dataset,
        "--det_net_path",
        args.det_net_path,
        "--det_net",
        args.det_net,
        "--ensemble_step",
        str(args.ensemble_step),
        "--inpainting_step",
        str(args.inpainting_step),
        "--effective_files",
        args.effective_files,
        "--n_patches",
        args.n_patches,
        "--savedir",
        str(savedir),
        "--save_scores",
        "--performance_det",
        "--performance",
    ]
    if args.robust_det_tta:
        cmd.extend(
            [
                "--robust_det_tta",
                "--robust_det_agg",
                args.robust_det_agg,
                "--robust_det_tta_variants",
                args.robust_det_tta_variants,
            ]
        )
    if clean_flag:
        cmd.append("--clean")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    elapsed = time.perf_counter() - t0
    mode_tag = "clean" if clean_flag else "attacked"
    log_path = savedir / f"realtime_saliuitl_{mode_tag}.log"
    log_path.write_text((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"saliuitl.py failed for {mode_tag}. See {log_path}")

    base = f"{savedir}_{args.dataset}_{args.det_net}_npatches_{args.n_patches}"
    suffix = "_clean" if clean_flag else ""
    scores = load_npy(Path(base + f"_ens_{args.ensemble_step}_scores{suffix}.npy")).reshape(-1)
    det_perfs = load_npy(Path(base + f"_ens_{args.ensemble_step}_perfs{suffix}.npy")).reshape(-1)
    det_clus = load_npy(Path(base + f"_ens_{args.ensemble_step}_clusperfs{suffix}.npy")).reshape(-1)
    rec_perfs = load_npy(Path(base + f"_inp_{args.inpainting_step}_perfs{suffix}.npy")).reshape(-1)
    rec_clus = load_npy(Path(base + f"_inp_{args.inpainting_step}_clusperfs{suffix}.npy")).reshape(-1)

    return {
        "elapsed_sec": elapsed,
        "scores": scores,
        "det_perfs": det_perfs,
        "det_clus": det_clus,
        "rec_perfs": rec_perfs,
        "rec_clus": rec_clus,
        "log_path": str(log_path),
    }


def simulate_realtime(run_data: dict, args) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(args.seed)
    scores = run_data["scores"]
    det_perfs = run_data["det_perfs"]
    rec_perfs = run_data["rec_perfs"]
    rec_clus = run_data["rec_clus"]

    threshold = det_threshold_from_args(args.det_mode, args.nn_det_threshold)
    period_ms = 1000.0 / args.target_fps

    rows = []
    rec_idx = 0
    prev_finish_ms = 0.0
    t_arrival_ms = 0.0
    dropped = 0

    for i in range(len(scores)):
        if i > 0:
            jitter = float(rng.normal(0.0, args.acq_jitter_ms))
            t_arrival_ms += max(0.0, period_ms + jitter)
        score = float(scores[i])
        detected = score >= threshold
        det_ms = float(det_perfs[i]) * 1000.0
        rec_ms = 0.0
        if detected:
            if rec_idx < len(rec_perfs):
                rec_ms = float(rec_perfs[rec_idx] + rec_clus[rec_idx]) * 1000.0
                rec_idx += 1
        proc_ms = det_ms + rec_ms

        acq_overhead = max(0.0, float(args.acq_overhead_ms + rng.normal(0.0, args.acq_overhead_jitter_ms)))
        acq_done_ms = t_arrival_ms + acq_overhead
        start_ms = max(acq_done_ms, prev_finish_ms)
        queue_wait_ms = start_ms - acq_done_ms
        finish_ms = start_ms + proc_ms
        latency_ms = finish_ms - t_arrival_ms
        deadline_miss = finish_ms > (t_arrival_ms + period_ms)

        drop = False
        if args.drop_threshold_ms > 0 and queue_wait_ms > args.drop_threshold_ms:
            drop = True
            dropped += 1
            # Skip heavy processing on dropped frame: maintain previous finish.
            finish_ms = prev_finish_ms
            latency_ms = queue_wait_ms

        rows.append(
            {
                "frame_idx": i,
                "arrival_ms": t_arrival_ms,
                "acq_overhead_ms": acq_overhead,
                "acq_done_ms": acq_done_ms,
                "start_proc_ms": start_ms,
                "finish_ms": finish_ms,
                "queue_wait_ms": queue_wait_ms,
                "latency_ms": latency_ms,
                "detected": int(detected),
                "score": score,
                "det_ms": det_ms,
                "rec_ms": rec_ms,
                "proc_ms": proc_ms,
                "deadline_miss": int(deadline_miss),
                "dropped": int(drop),
            }
        )
        prev_finish_ms = finish_ms

    valid_rows = [r for r in rows if r["dropped"] == 0]
    duration_ms = max(1.0, rows[-1]["finish_ms"] - rows[0]["arrival_ms"]) if rows else 1.0
    eff_fps = 1000.0 * len(valid_rows) / duration_ms
    miss_rate = float(sum(r["deadline_miss"] for r in rows)) / max(1, len(rows))
    det_rate = float(sum(r["detected"] for r in rows)) / max(1, len(rows))

    summary = {
        "frames_total": len(rows),
        "frames_processed": len(valid_rows),
        "frames_dropped": dropped,
        "target_fps": args.target_fps,
        "effective_fps": eff_fps,
        "detected_rate": det_rate,
        "deadline_miss_rate": miss_rate,
        "latency_ms_p50": pct([r["latency_ms"] for r in rows], 50),
        "latency_ms_p90": pct([r["latency_ms"] for r in rows], 90),
        "latency_ms_p95": pct([r["latency_ms"] for r in rows], 95),
        "latency_ms_p99": pct([r["latency_ms"] for r in rows], 99),
        "queue_wait_ms_p50": pct([r["queue_wait_ms"] for r in rows], 50),
        "queue_wait_ms_p90": pct([r["queue_wait_ms"] for r in rows], 90),
        "queue_wait_ms_p95": pct([r["queue_wait_ms"] for r in rows], 95),
        "proc_ms_p50": pct([r["proc_ms"] for r in rows], 50),
        "proc_ms_p90": pct([r["proc_ms"] for r in rows], 90),
        "proc_ms_p95": pct([r["proc_ms"] for r in rows], 95),
        "proc_ms_p99": pct([r["proc_ms"] for r in rows], 99),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Real-time style APRICOT pipeline for Saliuitl.")
    parser.add_argument("--python", default=".venv312/Scripts/python")
    parser.add_argument("--imgdir", default="data/apricot_saliuitl/clean")
    parser.add_argument("--patch_imgdir", default="data/apricot_saliuitl/1p")
    parser.add_argument("--dataset", default="voc")
    parser.add_argument("--det_net_path", default="checkpoints/final_detection/2dcnn_raw_VOC_5_atk_det.pth")
    parser.add_argument("--det_net", default="2dcnn_raw")
    parser.add_argument("--ensemble_step", type=int, default=5)
    parser.add_argument("--inpainting_step", type=int, default=5)
    parser.add_argument("--effective_files", default="effective_1p.npy")
    parser.add_argument("--n_patches", default="1")
    parser.add_argument("--det_mode", default="balanced", choices=("manual", "balanced", "attack_recall", "clean_speed"))
    parser.add_argument("--nn_det_threshold", type=float, default=0.5)
    parser.add_argument("--robust_det_tta", action="store_true")
    parser.add_argument("--robust_det_agg", default="max", choices=("max", "mean"))
    parser.add_argument("--robust_det_tta_variants", default="raw,blur,jpeg,gamma")
    parser.add_argument("--savedir", default="realtime_metrics")
    parser.add_argument("--target_fps", type=float, default=10.0)
    parser.add_argument("--acq_overhead_ms", type=float, default=8.0)
    parser.add_argument("--acq_overhead_jitter_ms", type=float, default=2.0)
    parser.add_argument("--acq_jitter_ms", type=float, default=3.0)
    parser.add_argument("--drop_threshold_ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_clean", action="store_true")
    args = parser.parse_args()

    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    attack_data = run_saliuitl(args, clean_flag=False)
    attack_rows, attack_summary = simulate_realtime(attack_data, args)
    write_csv(savedir / "realtime_attacked_per_frame.csv", attack_rows)

    summary = {
        "mode": "attacked",
        "saliuitl_elapsed_sec": attack_data["elapsed_sec"],
        "saliuitl_log": attack_data["log_path"],
        "realtime_sim": attack_summary,
        "config": vars(args),
    }

    if args.run_clean:
        clean_data = run_saliuitl(args, clean_flag=True)
        clean_rows, clean_summary = simulate_realtime(clean_data, args)
        write_csv(savedir / "realtime_clean_per_frame.csv", clean_rows)
        summary["clean_mode"] = {
            "saliuitl_elapsed_sec": clean_data["elapsed_sec"],
            "saliuitl_log": clean_data["log_path"],
            "realtime_sim": clean_summary,
        }

    (savedir / "realtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
