#!/usr/bin/env python3
"""
Analyze S0 (escenario negativo: 0 personas) from a metrics CSV.

Outputs:
- results/analysis_s0/summary_s0.csv
- results/analysis_s0/fig_s0_persons_vs_frame.png
- results/analysis_s0/fig_s0_fps_vs_frame.png
- results/analysis_s0/fig_s0_response_time_hist.png  (if response_time_ms exists)

Usage:
  python analyze_s0.py --metrics results/metrics.csv --out results/analysis_s0
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to metrics.csv")
    ap.add_argument("--out", default="results/analysis_s0", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.metrics)

    frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
    persons_col = pick_col(df, ["persons", "detected_persons", "person_count"])
    fps_col = pick_col(df, ["fps"])
    rt_col = pick_col(df, ["response_time_ms", "latency_ms", "time_ms"])
    avg_lm_col = pick_col(df, ["avg_landmarks", "avg_pose_landmarks"])

    missing = [("frame", frame_col), ("persons", persons_col), ("fps", fps_col)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise SystemExit(f"Faltan columnas en metrics.csv: {missing}. Columnas disponibles: {list(df.columns)}")

    # Core metrics
    N = len(df)
    persons = df[persons_col].astype(float)
    fps = df[fps_col].astype(float)

    # For S0, ground truth persons is 0 in every frame.
    # False positive event: persons > 0 when gt = 0.
    fp_frames = int((persons > 0).sum())
    fpr = fp_frames / N if N > 0 else float("nan")

    summary = {
        "frames_total": int(N),
        "false_positive_frames": int(fp_frames),
        "FPR": float(fpr),
        "persons_unique": ",".join(str(int(x)) for x in sorted(persons.unique())),
        "fps_mean": float(fps.mean()),
        "fps_min": float(fps.min()),
        "fps_max": float(fps.max()),
        "fps_cv": float(fps.std(ddof=1) / fps.mean()) if fps.mean() > 0 else float("nan"),
    }

    if rt_col is not None:
        rt = df[rt_col].astype(float)
        summary.update({
            "response_time_ms_mean": float(rt.mean()),
            "response_time_ms_p95": float(np.percentile(rt, 95)),
            "response_time_ms_min": float(rt.min()),
            "response_time_ms_max": float(rt.max()),
        })

    if avg_lm_col is not None:
        summary["avg_landmarks_mean"] = float(df[avg_lm_col].astype(float).mean())

    # Save summary CSV
    out_csv = os.path.join(args.out, "summary_s0.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    # Figures
    # 1) persons vs frame
    plt.figure(figsize=(10,4))
    plt.plot(df[frame_col], persons)
    plt.xlabel("Frame")
    plt.ylabel("Personas detectadas")
    plt.title("S0: conteo de personas por fotograma (escenario negativo)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "fig_s0_persons_vs_frame.png"), dpi=200)
    plt.close()

    # 2) fps vs frame
    plt.figure(figsize=(10,4))
    plt.plot(df[frame_col], fps)
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.title("S0: tasa de muestreo (FPS) por fotograma")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "fig_s0_fps_vs_frame.png"), dpi=200)
    plt.close()

    # 3) response time histogram (optional)
    if rt_col is not None:
        rt = df[rt_col].astype(float)
        p95 = float(np.percentile(rt, 95))
        plt.figure(figsize=(10,4))
        plt.hist(rt, bins=20)
        plt.axvline(p95, linewidth=2, label=f"p95 = {p95:.1f} ms")
        plt.xlabel("Tiempo de respuesta (ms)")
        plt.ylabel("Frecuencia")
        plt.title("S0: distribución del tiempo de respuesta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig_s0_response_time_hist.png"), dpi=200)
        plt.close()

    # Print quick summary
    print("=== Resumen S0 ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"Archivos generados en: {args.out}")

if __name__ == "__main__":
    main()
