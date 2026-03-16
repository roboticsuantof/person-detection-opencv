#!/usr/bin/env python3
"""
analyze_s2.py — análisis por réplica para escenario S2 (múltiples personas con entradas/salidas).

Genera en --out:
- summary_s2.csv
- fig_s2_persons_vs_frame.png  (incluye GT si se entrega --gt_file)
- fig_s2_fps_vs_frame.png
- fig_s2_response_time_hist.png (si existe response_time_ms)

Métricas S2 (post warm-up, comparando conteo estimado ĉ_i vs ground truth c_i por frame):
- Acc_S2: proporción de frames con ĉ_i = c_i
- R_under: proporción de frames con subconteo (ĉ_i < c_i)
- R_over: proporción de frames con sobreconteo (ĉ_i > c_i)
- MAE_count / RMSE_count: error absoluto medio / raíz del error cuadrático medio

Ground truth:
- Por frame: columnas (frame, gt_persons)
- Por segmentos: columnas (frame_start, frame_end, gt_persons)  [frame_end incluido]

NOTA: Se mantiene la línea vertical del warm-up, pero SIN leyenda (para evitar confusiones).

Uso ejemplo:
  python analyze_s2.py --metrics results/analysis_s2/S2_R1/metrics.csv --out results/analysis_s2/S2_R1 --skip_first 20 --gt_file results/ground_truth_s2.csv
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


def load_gt(gt_path: str, skip_first: int = 0) -> pd.DataFrame:
    """Soporta GT por frame o por segmentos; retorna frame, gt_persons (post warm-up)."""
    df = pd.read_csv(gt_path)

    if {'frame', 'gt_persons'}.issubset(df.columns):
        gt = df[['frame', 'gt_persons']].copy()

    elif {'frame_start', 'frame_end', 'gt_persons'}.issubset(df.columns):
        rows = []
        for _, r in df.iterrows():
            a = int(r['frame_start'])
            b = int(r['frame_end'])
            v = int(r['gt_persons'])
            if b < a:
                continue
            for f in range(a, b + 1):  # inclusive
                rows.append((f, v))
        gt = pd.DataFrame(rows, columns=['frame', 'gt_persons'])

    else:
        raise ValueError(
            'GT debe tener columnas (frame, gt_persons) o (frame_start, frame_end, gt_persons). '
            f'Columnas encontradas: {list(df.columns)}'
        )

    gt = gt.sort_values('frame').drop_duplicates('frame', keep='last')
    gt = gt[gt['frame'] >= int(skip_first)]
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', required=True, help='Path a metrics.csv')
    ap.add_argument('--out', required=True, help='Carpeta de salida (summary + figs)')
    ap.add_argument('--skip_first', type=int, default=20, help='Descartar primeros N frames (warm-up)')
    ap.add_argument('--gt_file', default=None, help='CSV ground truth (por frame o por segmentos). Opcional.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.metrics)

    frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
    persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
    fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
    rt_col = pick_col(df, ['response_time_ms', 'latency_ms', 'time_ms'])

    missing = [('frame', frame_col), ('persons', persons_col), ('fps', fps_col)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise SystemExit(
            f'Faltan columnas en metrics.csv: {missing}. '
            f'Columnas disponibles: {list(df.columns)}'
        )

    # Para plots mostramos todo; para stats usamos post warm-up
    df_plot = df.copy()
    df_stat = df.iloc[args.skip_first:].copy() if args.skip_first > 0 else df.copy()

    N_full = int(len(df_plot))
    N = int(len(df_stat))

    persons = df_stat[persons_col].astype(float).to_numpy()
    fps = df_stat[fps_col].astype(float).to_numpy()

    summary = {
        'scenario': 'S2',
        'frames_total': N_full,
        'frames_used': N,
        'skip_first': int(args.skip_first),
        'persons_unique_postwarmup': ','.join(str(int(x)) for x in sorted(np.unique(persons))) if N else '',
        'persons_mean_postwarmup': float(np.mean(persons)) if N else float('nan'),
        'persons_max_postwarmup': float(np.max(persons)) if N else float('nan'),
        'fps_mean': float(np.mean(fps)) if N else float('nan'),
        'fps_min': float(np.min(fps)) if N else float('nan'),
        'fps_max': float(np.max(fps)) if N else float('nan'),
        'fps_cv': float(np.std(fps, ddof=1) / np.mean(fps)) if N and np.mean(fps) > 0 else float('nan'),
    }

    # --- Métricas de conteo vs GT (si se entrega) ---
    gt_df = None
    if args.gt_file:
        gt_df = load_gt(args.gt_file, skip_first=args.skip_first)

        df_m = df_stat[[frame_col, persons_col]].copy()
        df_m.columns = ['frame', 'persons']

        df_eval = pd.merge(gt_df, df_m, on='frame', how='inner')

        if len(df_eval) == 0:
            summary.update({
                'gt_source': os.path.basename(args.gt_file),
                'frames_eval_gt': 0,
                'frames_correct': 0,
                'frames_under': 0,
                'frames_over': 0,
                'Acc_S2': float('nan'),
                'R_under': float('nan'),
                'R_over': float('nan'),
                'MAE_count': float('nan'),
                'RMSE_count': float('nan'),
            })
        else:
            c = df_eval['gt_persons'].astype(int).to_numpy()
            chat = df_eval['persons'].astype(int).to_numpy()
            diff = chat - c

            frames_eval = int(len(df_eval))
            frames_correct = int(np.sum(diff == 0))
            frames_under = int(np.sum(diff < 0))
            frames_over = int(np.sum(diff > 0))

            acc = frames_correct / frames_eval
            r_under = frames_under / frames_eval
            r_over = frames_over / frames_eval
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff.astype(float) ** 2)))

            summary.update({
                'gt_source': os.path.basename(args.gt_file),
                'frames_eval_gt': frames_eval,
                'frames_correct': frames_correct,
                'frames_under': frames_under,
                'frames_over': frames_over,
                'Acc_S2': float(acc),
                'R_under': float(r_under),
                'R_over': float(r_over),
                'MAE_count': mae,
                'RMSE_count': rmse,
            })

    # --- Métricas de latencia (si existe) ---
    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        summary.update({
            'response_time_ms_mean': float(np.mean(rt)) if N else float('nan'),
            'response_time_ms_p95': safe_percentile(rt, 95),
            'response_time_ms_min': float(np.min(rt)) if N else float('nan'),
            'response_time_ms_max': float(np.max(rt)) if N else float('nan'),
        })

    # Guardar summary
    pd.DataFrame([summary]).to_csv(os.path.join(args.out, 'summary_s2.csv'), index=False)

    # --- FIG 1: Persons vs frame (con GT si existe) ---
    plt.figure(figsize=(10, 4))
    plt.plot(df_plot[frame_col], df_plot[persons_col].astype(float), label='Conteo estimado')

    if args.gt_file:
        # Para visualización: reconstruimos GT "completo" (sin filtrar warm-up) para que se vea el perfil de entradas/salidas.
        gt_raw = pd.read_csv(args.gt_file)
        if {'frame', 'gt_persons'}.issubset(gt_raw.columns):
            gt_line = gt_raw[['frame', 'gt_persons']].copy()
        elif {'frame_start', 'frame_end', 'gt_persons'}.issubset(gt_raw.columns):
            rows = []
            for _, r in gt_raw.iterrows():
                a = int(r['frame_start']); b = int(r['frame_end']); v = int(r['gt_persons'])
                for f in range(a, b + 1):
                    rows.append((f, v))
            gt_line = pd.DataFrame(rows, columns=['frame', 'gt_persons'])
        else:
            gt_line = None

        if gt_line is not None and len(gt_line) > 0:
            gt_line = gt_line.sort_values('frame').drop_duplicates('frame', keep='last')
            plt.plot(gt_line['frame'], gt_line['gt_persons'].astype(float), linestyle='--', label='Ground truth')

    if 0 < args.skip_first < len(df_plot):
        x_cut = df_plot.iloc[args.skip_first][frame_col]
        plt.axvline(x_cut, linestyle='--', linewidth=2)  # SIN leyenda

    plt.xlabel('Frame')
    plt.ylabel('Personas')
    plt.title('S2: conteo de personas por fotograma')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'fig_s2_persons_vs_frame.png'), dpi=200)
    plt.close()

    # --- FIG 2: FPS vs frame ---
    plt.figure(figsize=(10, 4))
    plt.plot(df_plot[frame_col], df_plot[fps_col].astype(float))
    if 0 < args.skip_first < len(df_plot):
        x_cut = df_plot.iloc[args.skip_first][frame_col]
        plt.axvline(x_cut, linestyle='--', linewidth=2)  # SIN leyenda
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.title('S2: tasa de muestreo (FPS) por fotograma')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'fig_s2_fps_vs_frame.png'), dpi=200)
    plt.close()

    # --- FIG 3: Response time histogram (post warm-up) ---
    if rt_col is not None and rt_col in df_stat.columns:
        rt = df_stat[rt_col].astype(float).to_numpy()
        p95 = safe_percentile(rt, 95)
        plt.figure(figsize=(10, 4))
        plt.hist(rt, bins=20)
        plt.axvline(p95, linewidth=2, label=f'p95 = {p95:.1f} ms')
        plt.xlabel('Tiempo de respuesta (ms)')
        plt.ylabel('Frecuencia')
        plt.title('S2: distribución del tiempo de respuesta (post warm-up)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fig_s2_response_time_hist.png'), dpi=200)
        plt.close()

    # Print resumen en terminal
    print('=== RESUMEN S2 ===')
    for k, v in summary.items():
        print(f'{k:>28}: {v}')
    print(f'\nArchivos generados en: {args.out}')


if __name__ == '__main__':
    main()
