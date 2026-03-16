#!/usr/bin/env python3
"""
Combine 3 (or more) S0 replica analyses into one consolidated summary + optional overlay plots.

Expected structure (example):
  results/analysis_s0/S0_R1/summary_s0.csv
  results/analysis_s0/S0_R2/summary_s0.csv
  results/analysis_s0/S0_R3/summary_s0.csv

Optionally, if you also copy each replica's metrics.csv into its folder:
  results/analysis_s0/S0_R1/metrics.csv
  ...

Outputs:
  <root>/summary_s0_replicas.csv   (one row per replica)
  <root>/summary_s0_aggregate.csv  (aggregate: mean + worst-case)
  <root>/fig_s0_fps_overlay.png    (if metrics.csv exists for each replica)
  <root>/fig_s0_rt_overlay.png     (if metrics.csv exists for each replica)

Usage:
  python combine_s0_replicas.py --root results/analysis_s0 --pattern "S0_R*"

Notes:
- If you want to ignore warm-up, use --skip_first 20 to skip first N frames in each metrics.csv
  for the overlay plots and recomputed stats (mean/min/p95).
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/analysis_s0", help="Root directory containing replica folders")
    ap.add_argument("--pattern", default="S0_R*", help="Glob for replica folders under root")
    ap.add_argument("--skip_first", type=int, default=0, help="Skip first N frames when recomputing from metrics.csv")
    args = ap.parse_args()

    root = args.root
    rep_dirs = sorted(glob.glob(os.path.join(root, args.pattern)))
    if not rep_dirs:
        raise SystemExit(f"No replica folders found in {root} with pattern {args.pattern}")

    rows = []
    recomputed = []  # from metrics.csv when available

    for rd in rep_dirs:
        summary_path = os.path.join(rd, "summary_s0.csv")
        if not os.path.exists(summary_path):
            print(f"[WARN] Missing {summary_path}, skipping.")
            continue
        s = pd.read_csv(summary_path).iloc[0].to_dict()
        s["replica"] = os.path.basename(rd)
        rows.append(s)

        metrics_path = os.path.join(rd, "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            frame_col = "frame" if "frame" in df.columns else ("frame_idx" if "frame_idx" in df.columns else None)
            rt_col = "response_time_ms" if "response_time_ms" in df.columns else None
            fps_col = "fps" if "fps" in df.columns else None
            persons_col = "persons" if "persons" in df.columns else ("detected_persons" if "detected_persons" in df.columns else None)

            if frame_col and fps_col and persons_col:
                if args.skip_first > 0:
                    df = df.iloc[args.skip_first:].copy()

                fps = df[fps_col].astype(float).to_numpy()
                persons = df[persons_col].astype(float).to_numpy()

                rec = {
                    "replica": os.path.basename(rd),
                    "frames_used": int(len(df)),
                    "fps_mean_re": float(np.mean(fps)) if len(fps) else float("nan"),
                    "fps_min_re": float(np.min(fps)) if len(fps) else float("nan"),
                    "fps_max_re": float(np.max(fps)) if len(fps) else float("nan"),
                    "FPR_re": float(np.mean(persons > 0)) if len(persons) else float("nan"),
                }
                if rt_col and rt_col in df.columns:
                    rt = df[rt_col].astype(float).to_numpy()
                    rec.update({
                        "rt_mean_re": float(np.mean(rt)) if len(rt) else float("nan"),
                        "rt_p95_re": safe_percentile(rt, 95),
                        "rt_max_re": float(np.max(rt)) if len(rt) else float("nan"),
                    })
                recomputed.append(rec)

    if not rows:
        raise SystemExit("No summary_s0.csv files found to combine.")

    df_rep = pd.DataFrame(rows).sort_values("replica")
    out_rep = os.path.join(root, "summary_s0_replicas.csv")
    df_rep.to_csv(out_rep, index=False)

    agg = {}
    if recomputed:
        df_re = pd.DataFrame(recomputed).sort_values("replica")
        df_re.to_csv(os.path.join(root, "summary_s0_recomputed.csv"), index=False)

        agg["replicas_n"] = int(len(df_re))
        agg["skip_first"] = int(args.skip_first)
        agg["FPR_mean"] = float(df_re["FPR_re"].mean())
        agg["FPR_worst"] = float(df_re["FPR_re"].max())
        agg["fps_mean"] = float(df_re["fps_mean_re"].mean())
        agg["fps_min_worst"] = float(df_re["fps_min_re"].min())
        agg["fps_mean_worst"] = float(df_re["fps_mean_re"].min())

        if "rt_p95_re" in df_re.columns:
            agg["rt_p95_mean"] = float(df_re["rt_p95_re"].mean())
            agg["rt_p95_worst"] = float(df_re["rt_p95_re"].max())
            agg["rt_max_worst"] = float(df_re["rt_max_re"].max())
    else:
        agg["replicas_n"] = int(len(df_rep))
        agg["FPR_mean"] = float(df_rep["FPR"].mean())
        agg["FPR_worst"] = float(df_rep["FPR"].max())
        agg["fps_mean"] = float(df_rep["fps_mean"].mean())
        agg["fps_min_worst"] = float(df_rep["fps_min"].min())
        if "response_time_ms_p95" in df_rep.columns:
            agg["rt_p95_mean"] = float(df_rep["response_time_ms_p95"].mean())
            agg["rt_p95_worst"] = float(df_rep["response_time_ms_p95"].max())
            agg["rt_max_worst"] = float(df_rep["response_time_ms_max"].max())

    df_agg = pd.DataFrame([agg])
    out_agg = os.path.join(root, "summary_s0_aggregate.csv")
    df_agg.to_csv(out_agg, index=False)

    metrics_dirs = [rd for rd in rep_dirs if os.path.exists(os.path.join(rd, "metrics.csv"))]
    if len(metrics_dirs) >= 2:
        plt.figure(figsize=(10,4))
        for rd in metrics_dirs:
            df = pd.read_csv(os.path.join(rd, "metrics.csv"))
            frame_col = "frame" if "frame" in df.columns else ("frame_idx" if "frame_idx" in df.columns else None)
            if frame_col is None or "fps" not in df.columns:
                continue
            if args.skip_first > 0:
                df = df.iloc[args.skip_first:]
            plt.plot(df[frame_col], df["fps"].astype(float), label=os.path.basename(rd))
        plt.xlabel("Frame")
        plt.ylabel("FPS")
        plt.title("S0: FPS por fotograma (superposición de réplicas)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(root, "fig_s0_fps_overlay.png"), dpi=200)
        plt.close()

        has_rt = False
        for rd in metrics_dirs:
            dfh = pd.read_csv(os.path.join(rd, "metrics.csv"), nrows=5)
            if "response_time_ms" in dfh.columns:
                has_rt = True
                break
        if has_rt:
            plt.figure(figsize=(10,4))
            for rd in metrics_dirs:
                df = pd.read_csv(os.path.join(rd, "metrics.csv"))
                frame_col = "frame" if "frame" in df.columns else ("frame_idx" if "frame_idx" in df.columns else None)
                if frame_col is None or "response_time_ms" not in df.columns:
                    continue
                if args.skip_first > 0:
                    df = df.iloc[args.skip_first:]
                plt.plot(df[frame_col], df["response_time_ms"].astype(float), label=os.path.basename(rd))
            plt.xlabel("Frame")
            plt.ylabel("ms")
            plt.title("S0: tiempo de respuesta por fotograma (superposición de réplicas)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(root, "fig_s0_rt_overlay.png"), dpi=200)
            plt.close()

    print("OK")
    print(f"- Por réplica: {out_rep}")
    print(f"- Agregado: {out_agg}")
    if recomputed:
        print(f"- Recomputado desde metrics.csv (skip_first={args.skip_first}): {os.path.join(root,'summary_s0_recomputed.csv')}")
    if len(metrics_dirs) >= 2:
        print(f"- Figuras overlay en: {root}")

if __name__ == "__main__":
    main()
