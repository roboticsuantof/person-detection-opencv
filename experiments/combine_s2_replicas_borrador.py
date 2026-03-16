#!/usr/bin/env python3
"""
combine_s2_replicas.py — Consolida réplicas del escenario S2 (múltiples personas con entradas/salidas).

Estructura esperada:
  results/analysis_s2/S2_R1/metrics.csv
  results/analysis_s2/S2_R2/metrics.csv
  results/analysis_s2/S2_R3/metrics.csv

Opcional:
  results/analysis_s2/S2_Rx/summary_s2.csv (no es obligatorio; el script recomputa desde metrics)

Ground truth:
  --gt_file puede ser:
    - Por frame: columnas (frame, gt_persons)
    - Por segmentos: columnas (frame_start, frame_end, gt_persons)  [frame_end incluido]

Salidas (en --root):
  - summary_s2_recomputed.csv    (métricas por réplica, recomputadas)
  - summary_s2_aggregate.csv     (promedio y peor caso entre réplicas)
  - fig_s2_fps_overlay.png
  - fig_s2_rt_overlay.png        (si existe response_time_ms)
  - fig_s2_persons_overlay.png   (incluye GT si se entrega --gt_file)

Uso:
  python combine_s2_replicas.py --root results/analysis_s2 --pattern "S2_R*" --skip_first 20 --gt_file results/ground_truth_s2.csv
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
        return float('nan')
    return float(np.percentile(x, q))


def load_gt(gt_path: str) -> pd.DataFrame:
    """Carga GT por frame o por segmentos; retorna df con columnas frame, gt_persons."""

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
    return gt


def eval_metrics_vs_gt(df_metrics: pd.DataFrame, gt: pd.DataFrame, skip_first: int):
    """Evalúa conteo por frame vs GT y retorna dict con Acc_S2, R_under, R_over, MAE, RMSE y contadores."""

    frame_col = pick_col(df_metrics, ['frame', 'frame_idx', 'frame_id'])
    persons_col = pick_col(df_metrics, ['persons', 'detected_persons', 'person_count'])
    if frame_col is None or persons_col is None:
        return None

    dfm = df_metrics[[frame_col, persons_col]].copy()
    dfm.columns = ['frame', 'persons']
    if skip_first > 0:
        dfm = dfm[dfm['frame'] >= int(skip_first)]

    gt2 = gt.copy()
    if skip_first > 0:
        gt2 = gt2[gt2['frame'] >= int(skip_first)]

    dfe = pd.merge(gt2, dfm, on='frame', how='inner')
    if len(dfe) == 0:
        return {
            'frames_eval_gt': 0, 'frames_correct': 0, 'frames_under': 0, 'frames_over': 0,
            'Acc_S2': float('nan'), 'R_under': float('nan'), 'R_over': float('nan'),
            'MAE_count': float('nan'), 'RMSE_count': float('nan')
        }

    c = dfe['gt_persons'].astype(int).to_numpy()
    chat = dfe['persons'].astype(int).to_numpy()
    diff = chat - c

    N = int(len(dfe))
    correct = int(np.sum(diff == 0))
    under = int(np.sum(diff < 0))
    over = int(np.sum(diff > 0))
    acc = correct / N
    r_under = under / N
    r_over = over / N
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff.astype(float) ** 2)))

    return {
        'frames_eval_gt': N,
        'frames_correct': correct,
        'frames_under': under,
        'frames_over': over,
        'Acc_S2': float(acc),
        'R_under': float(r_under),
        'R_over': float(r_over),
        'MAE_count': mae,
        'RMSE_count': rmse
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/analysis_s2')
    ap.add_argument('--pattern', default='S2_R*')
    ap.add_argument('--skip_first', type=int, default=20)
    ap.add_argument('--gt_file', default=None, help='GT por frame o por segmentos (opcional, recomendado)')
    args = ap.parse_args()

    rep_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not rep_dirs:
        raise SystemExit(f'No se encontraron réplicas en {args.root} con patrón {args.pattern}')

    gt = load_gt(args.gt_file) if args.gt_file else None

    rows = []
    metrics_dirs = []
    for rd in rep_dirs:
        mp = os.path.join(rd, 'metrics.csv')
        if not os.path.exists(mp):
            continue
        metrics_dirs.append(rd)

        df = pd.read_csv(mp)
        frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
        persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
        fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
        rt_col = pick_col(df, ['response_time_ms', 'latency_ms', 'time_ms'])

        if frame_col is None or persons_col is None or fps_col is None:
            continue

        # stats post warm-up
        d = df.copy()
        if args.skip_first > 0:
            d = d[d[frame_col] >= int(args.skip_first)]

        persons = d[persons_col].astype(float).to_numpy()
        fps = d[fps_col].astype(float).to_numpy()
        N = int(len(d))

        rec = {
            'replica': os.path.basename(rd),
            'frames_used': N,
            'skip_first': int(args.skip_first),
            'persons_mean_postwarmup': float(np.mean(persons)) if N else float('nan'),
            'persons_max_postwarmup': float(np.max(persons)) if N else float('nan'),
            'fps_mean': float(np.mean(fps)) if N else float('nan'),
            'fps_min': float(np.min(fps)) if N else float('nan'),
            'fps_max': float(np.max(fps)) if N else float('nan'),
        }

        if rt_col is not None and rt_col in d.columns:
            rt = d[rt_col].astype(float).to_numpy()
            rec.update({
                'response_time_ms_mean': float(np.mean(rt)) if N else float('nan'),
                'response_time_ms_p95': safe_percentile(rt, 95),
                'response_time_ms_max': float(np.max(rt)) if N else float('nan'),
            })

        # evaluación vs GT (si hay)
        if gt is not None:
            ev = eval_metrics_vs_gt(df, gt, args.skip_first)
            rec.update(ev)
        rows.append(rec)

    if not rows:
        raise SystemExit('No hay metrics.csv válidos para recomputar en las réplicas.')

    df_re = pd.DataFrame(rows).sort_values('replica')
    df_re.to_csv(os.path.join(args.root, 'summary_s2_recomputed.csv'), index=False)

    # agregado (promedio y peor caso)
    agg = {
        'replicas_n': int(len(df_re)),
        'skip_first': int(args.skip_first),
        'fps_mean': float(df_re['fps_mean'].mean()),
        'fps_min_worst': float(df_re['fps_min'].min()),
    }
    if 'response_time_ms_p95' in df_re.columns:
        agg.update({
            'rt_p95_mean': float(df_re['response_time_ms_p95'].mean()),
            'rt_p95_worst': float(df_re['response_time_ms_p95'].max()),
            'rt_max_worst': float(df_re['response_time_ms_max'].max()),
        })
    if gt is not None and 'Acc_S2' in df_re.columns:
        agg.update({
            'Acc_mean': float(df_re['Acc_S2'].mean()),
            'Acc_worst': float(df_re['Acc_S2'].min()),
            'R_under_mean': float(df_re['R_under'].mean()),
            'R_under_worst': float(df_re['R_under'].max()),
            'R_over_mean': float(df_re['R_over'].mean()),
            'R_over_worst': float(df_re['R_over'].max()),
            'MAE_mean': float(df_re['MAE_count'].mean()),
            'MAE_worst': float(df_re['MAE_count'].max()),
        })

    pd.DataFrame([agg]).to_csv(os.path.join(args.root, 'summary_s2_aggregate.csv'), index=False)

    # overlays
    if len(metrics_dirs) >= 2:
        # FPS overlay
        plt.figure(figsize=(10,4))
        for rd in metrics_dirs:
            df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
            frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
            fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
            if args.skip_first > 0:
                df = df[df[frame_col] >= int(args.skip_first)]
            plt.plot(df[frame_col], df[fps_col].astype(float), label=os.path.basename(rd))
        plt.xlabel('Frame'); plt.ylabel('FPS')
        plt.title('S2: FPS por fotograma (superposición de réplicas)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, 'fig_s2_fps_overlay.png'), dpi=200)
        plt.close()

        # Persons overlay (con GT si hay)
        plt.figure(figsize=(10,4))
        for rd in metrics_dirs:
            df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
            frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
            persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
            if args.skip_first > 0:
                df = df[df[frame_col] >= int(args.skip_first)]
            plt.plot(df[frame_col], df[persons_col].astype(float), label=os.path.basename(rd))
        if gt is not None:
            gt_line = gt[gt['frame'] >= int(args.skip_first)].copy()
            if len(gt_line) > 0:
                plt.plot(gt_line['frame'], gt_line['gt_persons'].astype(float), linestyle='--', linewidth=2, label='Ground truth')
        plt.xlabel('Frame'); plt.ylabel('Personas')
        plt.title('S2: conteo por fotograma (superposición de réplicas)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, 'fig_s2_persons_overlay.png'), dpi=200)
        plt.close()

        # RT overlay (si existe)
        has_rt = any('response_time_ms' in pd.read_csv(os.path.join(rd, 'metrics.csv'), nrows=5).columns for rd in metrics_dirs)
        if has_rt:
            plt.figure(figsize=(10,4))
            for rd in metrics_dirs:
                df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
                frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
                if 'response_time_ms' not in df.columns:
                    continue
                if args.skip_first > 0:
                    df = df[df[frame_col] >= int(args.skip_first)]
                plt.plot(df[frame_col], df['response_time_ms'].astype(float), label=os.path.basename(rd))
            plt.xlabel('Frame'); plt.ylabel('ms')
            plt.title('S2: tiempo de respuesta por fotograma (superposición de réplicas)')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.root, 'fig_s2_rt_overlay.png'), dpi=200)
            plt.close()

    print('OK ->', args.root)


if __name__ == '__main__':
    main()
