#!/usr/bin/env python3
"""analyze_s0.py — versión con --skip_first (warm-up). Ver cabecera para detalles."""

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
        return float('nan')
    return float(np.percentile(x, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', required=True, help='Path a metrics.csv')
    ap.add_argument('--out', default=None, help='Carpeta de salida (default: carpeta del metrics)')
    ap.add_argument('--skip_first', type=int, default=0, help='Descartar primeros N frames (warm-up)')
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.dirname(args.metrics) or '.'
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.metrics)

    frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
    persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
    fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
    rt_col = pick_col(df, ['response_time_ms', 'latency_ms', 'time_ms'])

    missing = [name for name, col in [('frame', frame_col), ('persons', persons_col), ('fps', fps_col)] if col is None]
    if missing:
        raise SystemExit(f'Faltan columnas {missing} en metrics.csv. Columnas: {list(df.columns)}')

    df_plot = df.copy()
    df_stat = df.copy()
    if args.skip_first > 0:
        df_stat = df_stat[df_stat[frame_col] >= int(args.skip_first)]

    N_full = int(len(df_plot))
    N = int(len(df_stat))

    persons = df_stat[persons_col].astype(float).to_numpy()
    fps = df_stat[fps_col].astype(float).to_numpy()

    frames_fp = int(np.sum(persons > 0)) if N else 0
    fpr = frames_fp / N if N else float('nan')

    summary = {
        'scenario': 'S0',
        'frames_total': N_full,
        'frames_used': N,
        'skip_first': int(args.skip_first),
        'frames_with_persons': frames_fp,
        'FPR_S0': float(fpr),
        'fps_mean': float(np.mean(fps)) if N else float('nan'),
        'fps_min': float(np.min(fps)) if N else float('nan'),
        'fps_max': float(np.max(fps)) if N else float('nan'),
    }

    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        summary.update({
            'response_time_ms_mean': float(np.mean(rt)) if N else float('nan'),
            'response_time_ms_p95': safe_percentile(rt, 95),
            'response_time_ms_min': float(np.min(rt)) if N else float('nan'),
            'response_time_ms_max': float(np.max(rt)) if N else float('nan'),
        })

    pd.DataFrame([summary]).to_csv(os.path.join(args.out, 'summary_s0.csv'), index=False)

    # Persons vs frame
    plt.figure(figsize=(10,4))
    plt.plot(df_plot[frame_col], df_plot[persons_col].astype(float), label='Personas detectadas')
    if args.skip_first > 0:
        plt.axvline(int(args.skip_first), linestyle='--', linewidth=2)
    plt.xlabel('Frame'); plt.ylabel('Personas')
    plt.title('S0: personas detectadas por fotograma')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'fig_s0_persons_vs_frame.png'), dpi=200)
    plt.close()

    # FPS vs frame
    plt.figure(figsize=(10,4))
    plt.plot(df_plot[frame_col], df_plot[fps_col].astype(float))
    if args.skip_first > 0:
        plt.axvline(int(args.skip_first), linestyle='--', linewidth=2)
    plt.xlabel('Frame'); plt.ylabel('FPS')
    plt.title('S0: tasa de muestreo (FPS) por fotograma')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'fig_s0_fps_vs_frame.png'), dpi=200)
    plt.close()

    # Latency histogram
    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        p95 = safe_percentile(rt, 95)
        plt.figure(figsize=(10,4))
        plt.hist(rt, bins=20)
        plt.axvline(p95, linewidth=2, label=f'p95 = {p95:.1f} ms')
        plt.xlabel('Tiempo de respuesta (ms)')
        plt.ylabel('Frecuencia')
        plt.title('S0: distribución del tiempo de respuesta (post warm-up)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fig_s0_response_time_hist.png'), dpi=200)
        plt.close()

    print('=== RESUMEN S0 ===')
    for k, v in summary.items():
        print(f'{k:>24}: {v}')
    print(f'\nArchivos generados en: {args.out}')


if __name__ == '__main__':
    main()
