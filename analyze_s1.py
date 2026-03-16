#!/usr/bin/env python3
"""
analyze_s1.py — análisis por réplica para escenario S1 (1 sujeto con posibles entradas/salidas).

Soporta:
  --gt <int>            Ground truth constante (ej: 1)
  --gt_file <path.csv>  Ground truth por frame o por segmentos (recomendado si hay entradas/salidas)

Formato GT aceptado:
  - Por frame: columnas (frame, gt_persons)
  - Por segmentos: columnas (frame_start, frame_end, gt_persons)   [frame_end incluido]

Argumentos:
  --metrics     metrics.csv (requerido)
  --out         carpeta salida (requerido)
  --skip_first  descarta primeros N frames (warm-up) (default: 0)
  --gt          gt constante (opcional)
  --gt_file     archivo GT (opcional)

Genera en --out:
  - summary_s1.csv
  - fig_s1_persons_vs_frame.png   (incluye GT si está disponible)
  - fig_s1_fps_vs_frame.png
  - fig_s1_response_time_hist.png (si existe response_time_ms)

Métricas (post warm-up):
  - Acc_S1_exact: % frames donde persons == gt_persons (exacto)
  - FNR_miss:     % frames con gt=1 y persons==0
  - FPR_extra:    % frames con gt=0 y persons>0
  - Overcount:    % frames con gt=1 y persons>1
  - FPS mean/min/max, RT mean/p95/max (si existe)
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
        return float('nan')
    return float(np.percentile(x, q))


def load_gt(gt_path: str) -> pd.DataFrame:
    df = pd.read_csv(gt_path)

    if {'frame', 'gt_persons'}.issubset(df.columns):
        gt = df[['frame', 'gt_persons']].copy()

    elif {'frame_start', 'frame_end', 'gt_persons'}.issubset(df.columns):
        rows = []
        for _, r in df.iterrows():
            a = int(r['frame_start']); b = int(r['frame_end']); v = int(r['gt_persons'])
            if b < a:
                continue
            for f in range(a, b + 1):  # inclusive
                rows.append((f, v))
        gt = pd.DataFrame(rows, columns=['frame', 'gt_persons'])
    else:
        raise ValueError(
            'GT debe tener (frame, gt_persons) o (frame_start, frame_end, gt_persons). '
            f'Columnas encontradas: {list(df.columns)}'
        )

    gt = gt.sort_values('frame').drop_duplicates('frame', keep='last')
    gt['frame'] = gt['frame'].astype(int)
    gt['gt_persons'] = gt['gt_persons'].astype(int)
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip_first', type=int, default=0)

    # compatibilidad: --gt (constante) o --gt_file (archivo)
    ap.add_argument('--gt', type=int, default=None, help='GT constante (ej: 1). Alternativa a --gt_file.')
    ap.add_argument('--gt_file', default=None, help='GT por frame/segmentos (CSV). Alternativa a --gt.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.metrics)

    frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
    persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
    fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
    rt_col = pick_col(df, ['response_time_ms', 'latency_ms', 'time_ms'])

    missing = [name for name, col in [('frame', frame_col), ('persons', persons_col), ('fps', fps_col)] if col is None]
    if missing:
        raise SystemExit(f'Faltan columnas {missing} en metrics.csv. Columnas: {list(df.columns)}')

    # cargar GT si aplica
    gt_df = None
    gt_source = ''
    if args.gt_file:
        gt_df = load_gt(args.gt_file)
        gt_source = os.path.basename(args.gt_file)
    elif args.gt is not None:
        gt_df = pd.DataFrame({'frame': df[frame_col].astype(int), 'gt_persons': int(args.gt)})
        gt_source = f'constant:{int(args.gt)}'

    # separar plot y stats post warm-up
    df_plot = df.copy()
    df_stat = df.copy()
    if args.skip_first > 0:
        df_stat = df_stat[df_stat[frame_col] >= int(args.skip_first)]

    N_total = int(len(df_plot))
    N = int(len(df_stat))

    persons = df_stat[persons_col].astype(int).to_numpy()
    fps = df_stat[fps_col].astype(float).to_numpy()

    summary = {
        'scenario': 'S1',
        'frames_total': N_total,
        'frames_used': N,
        'skip_first': int(args.skip_first),
        'gt_source': gt_source,
        'fps_mean': float(np.mean(fps)) if N else float('nan'),
        'fps_min': float(np.min(fps)) if N else float('nan'),
        'fps_max': float(np.max(fps)) if N else float('nan'),
    }

    # métricas vs GT (si existe)
    if gt_df is not None:
        m = df_stat[[frame_col, persons_col]].copy()
        m.columns = ['frame', 'persons']
        m['frame'] = m['frame'].astype(int)

        gt2 = gt_df.copy()
        if args.skip_first > 0:
            gt2 = gt2[gt2['frame'] >= int(args.skip_first)]

        dfe = pd.merge(gt2, m, on='frame', how='inner')
        if len(dfe) == 0:
            summary.update({
                'frames_eval_gt': 0,
                'Acc_S1_exact': float('nan'),
                'FNR_miss': float('nan'),
                'FPR_extra': float('nan'),
                'Overcount': float('nan'),
            })
        else:
            g = dfe['gt_persons'].astype(int).to_numpy()
            p = dfe['persons'].astype(int).to_numpy()

            acc = float(np.mean(p == g))
            miss = float(np.mean((g == 1) & (p == 0)))
            extra = float(np.mean((g == 0) & (p > 0)))
            overc = float(np.mean((g == 1) & (p > 1)))

            summary.update({
                'frames_eval_gt': int(len(dfe)),
                'Acc_S1_exact': acc,
                'FNR_miss': miss,
                'FPR_extra': extra,
                'Overcount': overc,
            })

    # tiempo de respuesta (si está)
    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        summary.update({
            'response_time_ms_mean': float(np.mean(rt)) if N else float('nan'),
            'response_time_ms_p95': safe_percentile(rt, 95),
            'response_time_ms_max': float(np.max(rt)) if N else float('nan'),
        })

    pd.DataFrame([summary]).to_csv(os.path.join(args.out, 'summary_s1.csv'), index=False)

    # persons vs frame
    plt.figure(figsize=(10,4))
    plt.plot(df_plot[frame_col], df_plot[persons_col].astype(float), label='Estimación (persons)')
    if gt_df is not None:
        plt.plot(gt_df['frame'], gt_df['gt_persons'].astype(float), linestyle='--', linewidth=2, label='GT')
    if args.skip_first > 0:
        plt.axvline(int(args.skip_first), linestyle='--', linewidth=2)
    plt.xlabel('Frame'); plt.ylabel('Personas')
    plt.title('S1: conteo de personas por fotograma')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'fig_s1_persons_vs_frame.png'), dpi=200)
    plt.close()

    # fps vs frame
    plt.figure(figsize=(10,4))
    plt.plot(df_plot[frame_col], df_plot[fps_col].astype(float))
    if args.skip_first > 0:
        plt.axvline(int(args.skip_first), linestyle='--', linewidth=2)
    plt.xlabel('Frame'); plt.ylabel('FPS')
    plt.title('S1: tasa de muestreo (FPS) por fotograma')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'fig_s1_fps_vs_frame.png'), dpi=200)
    plt.close()

    # hist RT
    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        p95 = safe_percentile(rt, 95)
        plt.figure(figsize=(10,4))
        plt.hist(rt, bins=20)
        plt.axvline(p95, linewidth=2, label=f'p95 = {p95:.1f} ms')
        plt.xlabel('Tiempo de respuesta (ms)')
        plt.ylabel('Frecuencia')
        plt.title('S1: distribución del tiempo de respuesta (post warm-up)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fig_s1_response_time_hist.png'), dpi=200)
        plt.close()

    print('=== RESUMEN S1 ===')
    for k, v in summary.items():
        print(f'{k:>24}: {v}')
    print(f'\nArchivos generados en: {args.out}')


if __name__ == '__main__':
    main()
