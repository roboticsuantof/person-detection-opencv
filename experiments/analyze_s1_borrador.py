#!/usr/bin/env python3
"""
analyze_s1.py — análisis por réplica para escenario S1 (1 persona).

Genera en --out:
- summary_s1.csv
- fig_s1_persons_vs_frame.png
- fig_s1_fps_vs_frame.png
- fig_s1_response_time_hist.png (si existe response_time_ms)

Métricas S1 (post warm-up):
- Acc_S1: proporción de frames con persons == gt (por defecto gt=1)
- FNR_miss: proporción de frames con persons < gt (típicamente 0 cuando gt=1)
- FPR_extra: proporción de frames con persons > gt
- MAE_count / RMSE_count: error respecto a gt

NOTA: Se mantiene la línea vertical del warm-up, pero SIN leyenda (para evitar confusiones).

Uso:
  python analyze_s1.py --metrics results/analysis_s1/S1_R1/metrics.csv --out results/analysis_s1/S1_R1 --skip_first 20 --gt 1
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


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path a metrics.csv")
    ap.add_argument("--out", required=True, help="Carpeta de salida (summary + figs)")
    ap.add_argument("--skip_first", type=int, default=20, help="Descartar primeros N frames (warm-up)")
    ap.add_argument("--gt", type=int, default=1, help="Ground truth de personas para S1 (default=1)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.metrics)

    frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
    persons_col = pick_col(df, ["persons", "detected_persons", "person_count"])
    fps_col = pick_col(df, ["fps", "sampling_rate_hz"])
    rt_col = pick_col(df, ["response_time_ms", "latency_ms", "time_ms"])

    missing = [("frame", frame_col), ("persons", persons_col), ("fps", fps_col)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise SystemExit(
            f"Faltan columnas en metrics.csv: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Para plots mostramos todo; para stats usamos post warm-up
    df_plot = df.copy()
    df_stat = df.iloc[args.skip_first:].copy() if args.skip_first > 0 else df.copy()

    N_full = int(len(df_plot))
    N = int(len(df_stat))

    persons = df_stat[persons_col].astype(float).to_numpy()
    fps = df_stat[fps_col].astype(float).to_numpy()
    gt = float(args.gt)

    # Métricas S1 (conteo vs gt)
    acc = float(np.mean(persons == gt)) if N else float("nan")
    fnr_miss = float(np.mean(persons < gt)) if N else float("nan")
    fpr_extra = float(np.mean(persons > gt)) if N else float("nan")
    mae = float(np.mean(np.abs(persons - gt))) if N else float("nan")
    rmse = float(np.sqrt(np.mean((persons - gt) ** 2))) if N else float("nan")

    summary = {
        "scenario": "S1",
        "gt_persons": int(args.gt),
        "frames_total": N_full,
        "frames_used": N,
        "skip_first": int(args.skip_first),
        "Acc_S1": acc,
        "FNR_miss": fnr_miss,
        "FPR_extra": fpr_extra,
        "MAE_count": mae,
        "RMSE_count": rmse,
        "persons_unique": ",".join(str(int(x)) for x in sorted(np.unique(persons))) if N else "",
        "fps_mean": float(np.mean(fps)) if N else float("nan"),
        "fps_min": float(np.min(fps)) if N else float("nan"),
        "fps_max": float(np.max(fps)) if N else float("nan"),
        "fps_cv": float(np.std(fps, ddof=1) / np.mean(fps)) if N and np.mean(fps) > 0 else float("nan"),
    }

    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        summary.update({
            "response_time_ms_mean": float(np.mean(rt)) if N else float("nan"),
            "response_time_ms_p95": safe_percentile(rt, 95),
            "response_time_ms_min": float(np.min(rt)) if N else float("nan"),
            "response_time_ms_max": float(np.max(rt)) if N else float("nan"),
        })

    # Guardar summary
    pd.DataFrame([summary]).to_csv(os.path.join(args.out, "summary_s1.csv"), index=False)

    # --- FIG 1: Persons vs frame ---
    plt.figure(figsize=(10, 4))
    plt.plot(df_plot[frame_col], df_plot[persons_col].astype(float))
    if 0 < args.skip_first < len(df_plot):
        x_cut = df_plot.iloc[args.skip_first][frame_col]
        plt.axvline(x_cut, linestyle="--", linewidth=2)  # SIN leyenda
    plt.xlabel("Frame")
    plt.ylabel("Personas detectadas")
    plt.title("S1: conteo de personas por fotograma")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "fig_s1_persons_vs_frame.png"), dpi=200)
    plt.close()

    # --- FIG 2: FPS vs frame ---
    plt.figure(figsize=(10, 4))
    plt.plot(df_plot[frame_col], df_plot[fps_col].astype(float))
    if 0 < args.skip_first < len(df_plot):
        x_cut = df_plot.iloc[args.skip_first][frame_col]
        plt.axvline(x_cut, linestyle="--", linewidth=2)  # SIN leyenda
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.title("S1: tasa de muestreo (FPS) por fotograma")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "fig_s1_fps_vs_frame.png"), dpi=200)
    plt.close()

    # --- FIG 3: Response time histogram (post warm-up) ---
    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        p95 = safe_percentile(rt, 95)
        plt.figure(figsize=(10, 4))
        plt.hist(rt, bins=20)
        plt.axvline(p95, linewidth=2, label=f"p95 = {p95:.1f} ms")
        plt.xlabel("Tiempo de respuesta (ms)")
        plt.ylabel("Frecuencia")
        plt.title("S1: distribución del tiempo de respuesta (post warm-up)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig_s1_response_time_hist.png"), dpi=200)
        plt.close()

    # Print resumen en terminal (estilo analyze_s0)
    print("=== RESUMEN S1 ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")
    print(f"\nArchivos generados en: {args.out}")


if __name__ == "__main__":
    main()