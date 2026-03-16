#!/usr/bin/env python3
"""
Combine S1 replica analyses into one consolidated summary + overlay plots.

Expected structure:
  results/analysis_s1/S1_R1/summary_s1.csv
  results/analysis_s1/S1_R2/summary_s1.csv
  results/analysis_s1/S1_R3/summary_s1.csv

And (recommended):
  results/analysis_s1/S1_Rx/metrics.csv

Outputs (in --root):
  summary_s1_replicas.csv
  summary_s1_aggregate.csv
  summary_s1_recomputed.csv
  fig_s1_fps_overlay.png
  fig_s1_rt_overlay.png
  fig_s1_persons_overlay.png

Usage:
  python combine_s1_replicas.py --root results/analysis_s1 --pattern "S1_R*" --skip_first 20
"""

import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/analysis_s1")
    ap.add_argument("--pattern", default="S1_R*")
    ap.add_argument("--skip_first", type=int, default=20)
    ap.add_argument("--gt", type=int, default=1)
    args = ap.parse_args()

    rep_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not rep_dirs:
        raise SystemExit(f"No se encontraron réplicas en {args.root} con patrón {args.pattern}")

    # 1) summary_s1.csv por réplica (si existe)
    rows = []
    for rd in rep_dirs:
        sp = os.path.join(rd, "summary_s1.csv")
        if os.path.exists(sp):
            d = pd.read_csv(sp).iloc[0].to_dict()
            d["replica"] = os.path.basename(rd)
            rows.append(d)
    if rows:
        pd.DataFrame(rows).sort_values("replica").to_csv(os.path.join(args.root, "summary_s1_replicas.csv"), index=False)

    # 2) recomputar desde metrics.csv (para agregados consistentes)
    recomputed, metrics_dirs = [], []
    for rd in rep_dirs:
        mp = os.path.join(rd, "metrics.csv")
        if not os.path.exists(mp):
            continue
        metrics_dirs.append(rd)

        df = pd.read_csv(mp)
        frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
        persons_col = pick_col(df, ["persons", "detected_persons", "person_count"])
        fps_col = pick_col(df, ["fps", "sampling_rate_hz"])
        rt_col = pick_col(df, ["response_time_ms", "latency_ms", "time_ms"])

        if frame_col is None or persons_col is None or fps_col is None:
            continue

        if args.skip_first > 0:
            df = df.iloc[args.skip_first:].copy()

        persons = df[persons_col].astype(float).to_numpy()
        fps = df[fps_col].astype(float).to_numpy()
        gt = float(args.gt)
        N = int(len(df))

        acc = float(np.mean(persons == gt)) if N else float("nan")
        fnr_miss = float(np.mean(persons < gt)) if N else float("nan")
        fpr_extra = float(np.mean(persons > gt)) if N else float("nan")
        mae = float(np.mean(np.abs(persons - gt))) if N else float("nan")
        rmse = float(np.sqrt(np.mean((persons - gt) ** 2))) if N else float("nan")

        rec = {
            "replica": os.path.basename(rd),
            "frames_used": N,
            "skip_first": int(args.skip_first),
            "gt_persons": int(args.gt),
            "Acc_S1": acc,
            "FNR_miss": fnr_miss,
            "FPR_extra": fpr_extra,
            "MAE_count": mae,
            "RMSE_count": rmse,
            "fps_mean": float(np.mean(fps)) if N else float("nan"),
            "fps_min": float(np.min(fps)) if N else float("nan"),
            "fps_max": float(np.max(fps)) if N else float("nan"),
        }

        if rt_col is not None and rt_col in df.columns:
            rt = df[rt_col].astype(float).to_numpy()
            rec.update({
                "response_time_ms_mean": float(np.mean(rt)) if N else float("nan"),
                "response_time_ms_p95": safe_percentile(rt, 95),
                "response_time_ms_max": float(np.max(rt)) if N else float("nan"),
            })

        recomputed.append(rec)

    if not recomputed:
        raise SystemExit("No hay metrics.csv válidos para recomputar en las réplicas.")

    df_re = pd.DataFrame(recomputed).sort_values("replica")
    df_re.to_csv(os.path.join(args.root, "summary_s1_recomputed.csv"), index=False)

    agg = {
        "replicas_n": int(len(df_re)),
        "skip_first": int(args.skip_first),
        "gt_persons": int(args.gt),
        "Acc_mean": float(df_re["Acc_S1"].mean()),
        "Acc_worst": float(df_re["Acc_S1"].min()),
        "FNR_miss_mean": float(df_re["FNR_miss"].mean()),
        "FNR_miss_worst": float(df_re["FNR_miss"].max()),
        "FPR_extra_mean": float(df_re["FPR_extra"].mean()),
        "FPR_extra_worst": float(df_re["FPR_extra"].max()),
        "MAE_mean": float(df_re["MAE_count"].mean()),
        "RMSE_mean": float(df_re["RMSE_count"].mean()),
        "fps_mean": float(df_re["fps_mean"].mean()),
        "fps_min_worst": float(df_re["fps_min"].min()),
    }
    if "response_time_ms_p95" in df_re.columns:
        agg.update({
            "rt_p95_mean": float(df_re["response_time_ms_p95"].mean()),
            "rt_p95_worst": float(df_re["response_time_ms_p95"].max()),
            "rt_max_worst": float(df_re["response_time_ms_max"].max()),
        })

    pd.DataFrame([agg]).to_csv(os.path.join(args.root, "summary_s1_aggregate.csv"), index=False)

    # 3) overlays (si hay >=2 réplicas con metrics.csv)
    if len(metrics_dirs) >= 2:
        # FPS overlay
        plt.figure(figsize=(10,4))
        for rd in metrics_dirs:
            df = pd.read_csv(os.path.join(rd, "metrics.csv"))
            frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
            fps_col = pick_col(df, ["fps", "sampling_rate_hz"])
            if args.skip_first > 0:
                df = df.iloc[args.skip_first:]
            plt.plot(df[frame_col], df[fps_col].astype(float), label=os.path.basename(rd))
        plt.xlabel("Frame"); plt.ylabel("FPS")
        plt.title("S1: FPS por fotograma (superposición de réplicas)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, "fig_s1_fps_overlay.png"), dpi=200)
        plt.close()

        # Persons overlay
        plt.figure(figsize=(10,4))
        for rd in metrics_dirs:
            df = pd.read_csv(os.path.join(rd, "metrics.csv"))
            frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
            persons_col = pick_col(df, ["persons", "detected_persons", "person_count"])
            if args.skip_first > 0:
                df = df.iloc[args.skip_first:]
            plt.plot(df[frame_col], df[persons_col].astype(float), label=os.path.basename(rd))
        plt.xlabel("Frame"); plt.ylabel("Personas detectadas")
        plt.title("S1: conteo por fotograma (superposición de réplicas)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, "fig_s1_persons_overlay.png"), dpi=200)
        plt.close()

        # RT overlay (si existe)
        has_rt = any("response_time_ms" in pd.read_csv(os.path.join(rd, "metrics.csv"), nrows=5).columns for rd in metrics_dirs)
        if has_rt:
            plt.figure(figsize=(10,4))
            for rd in metrics_dirs:
                df = pd.read_csv(os.path.join(rd, "metrics.csv"))
                frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
                if "response_time_ms" not in df.columns:
                    continue
                if args.skip_first > 0:
                    df = df.iloc[args.skip_first:]
                plt.plot(df[frame_col], df["response_time_ms"].astype(float), label=os.path.basename(rd))
            plt.xlabel("Frame"); plt.ylabel("ms")
            plt.title("S1: tiempo de respuesta por fotograma (superposición de réplicas)")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.root, "fig_s1_rt_overlay.png"), dpi=200)
            plt.close()

    print("OK ->", args.root)

if __name__ == "__main__":
    main()