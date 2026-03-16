#!/usr/bin/env python3
"""
combine_s2_replicas_v2.py — Consolida réplicas del escenario S2 (múltiples personas con entradas/salidas).

Busca réplicas en:
  --root / --pattern   (ej: results/analysis_s2 / "S2_R*")

Por cada réplica espera:
  - metrics.csv  (requerido)
  - summary_s2.csv (opcional; si existe se consolidará en summary_s2_replicas.csv)
  - ground_truth.csv (opcional, por réplica si usas --gt_name)

Ground truth soportado:
  - Por réplica: --gt_name ground_truth.csv (default)  -> busca ese nombre dentro de cada carpeta S2_Rx
  - Global:      --gt_global <path.csv>               -> mismo GT para todas (solo si aplica)
  - Sin GT:      no calcula Acc/MAE/R_under/R_over

Formato GT aceptado:
  - Por frame:    columnas (frame, gt_persons)
  - Por segmentos: columnas (frame_start, frame_end, gt_persons) [frame_end incluido]

Salidas (en --root):
  - summary_s2_replicas.csv     (si encuentra summary_s2.csv en réplicas)
  - summary_s2_recomputed.csv   (métricas por réplica recomputadas desde metrics (+ GT si hay))
  - summary_s2_aggregate.csv    (promedio y peor caso)
  - fig_s2_fps_overlay.png
  - fig_s2_persons_overlay.png
  - fig_s2_rt_overlay.png       (si existe response_time_ms)

Uso típico (GT por réplica):
  python combine_s2_replicas.py --root results/analysis_s2 --pattern "S2_R*" --skip_first 20 --gt_name ground_truth.csv
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
            f'GT inválido: {gt_path}. Debe tener (frame, gt_persons) o (frame_start, frame_end, gt_persons). '
            f'Columnas encontradas: {list(df.columns)}'
        )

    gt = gt.sort_values('frame').drop_duplicates('frame', keep='last')
    gt['frame'] = gt['frame'].astype(int)
    gt['gt_persons'] = gt['gt_persons'].astype(int)
    return gt


def eval_metrics_vs_gt(df_metrics: pd.DataFrame, gt: pd.DataFrame, skip_first: int):
    """Evalúa conteo por frame vs GT (S2)."""
    frame_col = pick_col(df_metrics, ['frame', 'frame_idx', 'frame_id'])
    persons_col = pick_col(df_metrics, ['persons', 'detected_persons', 'person_count'])
    if frame_col is None or persons_col is None:
        return None

    dfm = df_metrics[[frame_col, persons_col]].copy()
    dfm.columns = ['frame', 'persons']
    dfm['frame'] = dfm['frame'].astype(int)
    dfm['persons'] = dfm['persons'].astype(int)

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

    return {
        'frames_eval_gt': N,
        'frames_correct': correct,
        'frames_under': under,
        'frames_over': over,
        'Acc_S2': float(correct / N),
        'R_under': float(under / N),
        'R_over': float(over / N),
        'MAE_count': float(np.mean(np.abs(diff))),
        'RMSE_count': float(np.sqrt(np.mean(diff.astype(float) ** 2))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/analysis_s2')
    ap.add_argument('--pattern', default='S2_R*')
    ap.add_argument('--skip_first', type=int, default=20)

    ap.add_argument('--gt_global', default=None, help='Ruta a GT global (mismo para todas).')
    ap.add_argument('--gt_name', default='ground_truth.csv', help='Nombre de GT por réplica dentro de cada carpeta.')
    args = ap.parse_args()

    rep_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not rep_dirs:
        raise SystemExit(f'No se encontraron réplicas en {args.root} con patrón {args.pattern}')

    # 1) Consolidar summary_s2.csv (si existen)
    summaries = []
    for rd in rep_dirs:
        sp = os.path.join(rd, 'summary_s2.csv')
        if os.path.exists(sp):
            s = pd.read_csv(sp)
            if len(s):
                s.insert(0, 'replica', os.path.basename(rd))
                summaries.append(s)
    if summaries:
        df_sum = pd.concat(summaries, ignore_index=True)
        df_sum.to_csv(os.path.join(args.root, 'summary_s2_replicas.csv'), index=False)

    # 2) Preparar GT global si aplica
    gt_global_df = load_gt(args.gt_global) if args.gt_global else None

    rows = []
    metrics_dirs = []
    has_rt_any = False

    for rd in rep_dirs:
        mp = os.path.join(rd, 'metrics.csv')
        if not os.path.exists(mp):
            continue

        df = pd.read_csv(mp)
        frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
        persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
        fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
        rt_col = pick_col(df, ['response_time_ms', 'latency_ms', 'time_ms'])

        if frame_col is None or persons_col is None or fps_col is None:
            continue

        metrics_dirs.append(rd)

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
            has_rt_any = True
            rt = d[rt_col].astype(float).to_numpy()
            rec.update({
                'response_time_ms_mean': float(np.mean(rt)) if N else float('nan'),
                'response_time_ms_p95': safe_percentile(rt, 95),
                'response_time_ms_max': float(np.max(rt)) if N else float('nan'),
            })

        # Evaluación GT (global o por réplica)
        gt_df = None
        gt_src = ''
        if gt_global_df is not None:
            gt_df = gt_global_df
            gt_src = os.path.basename(args.gt_global)
        else:
            gp = os.path.join(rd, args.gt_name)
            if os.path.exists(gp):
                gt_df = load_gt(gp)
                gt_src = args.gt_name

        if gt_df is not None:
            rec['gt_source'] = gt_src
            rec.update(eval_metrics_vs_gt(df, gt_df, args.skip_first))
        else:
            rec['gt_source'] = ''

        rows.append(rec)

    if not rows:
        raise SystemExit('No hay metrics.csv válidos para recomputar en las réplicas.')

    df_re = pd.DataFrame(rows).sort_values('replica')
    df_re.to_csv(os.path.join(args.root, 'summary_s2_recomputed.csv'), index=False)

    # 3) Agregado
    agg = {
        'replicas_n': int(len(df_re)),
        'skip_first': int(args.skip_first),
        'fps_mean': float(df_re['fps_mean'].mean()),
        'fps_min_worst': float(df_re['fps_min'].min()),
    }

    if has_rt_any and 'response_time_ms_p95' in df_re.columns:
        agg.update({
            'rt_p95_mean': float(df_re['response_time_ms_p95'].mean()),
            'rt_p95_worst': float(df_re['response_time_ms_p95'].max()),
            'rt_max_worst': float(df_re['response_time_ms_max'].max()),
        })

    if 'Acc_S2' in df_re.columns and df_re['Acc_S2'].notna().any():
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

    # 4) Overlays (si hay 2+ métricas)
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

        # Persons overlay (y GT global si existe)
        plt.figure(figsize=(10,4))
        for rd in metrics_dirs:
            df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
            frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
            persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
            if args.skip_first > 0:
                df = df[df[frame_col] >= int(args.skip_first)]
            plt.plot(df[frame_col], df[persons_col].astype(float), label=os.path.basename(rd))

        # Si hay GT global, lo dibujamos como referencia
        if gt_global_df is not None:
            gt_line = gt_global_df[gt_global_df['frame'] >= int(args.skip_first)].copy()
            if len(gt_line) > 0:
                plt.plot(gt_line['frame'], gt_line['gt_persons'].astype(float), linestyle='--', linewidth=2, label='Ground truth (global)')

        plt.xlabel('Frame'); plt.ylabel('Personas')
        plt.title('S2: conteo por fotograma (superposición de réplicas)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, 'fig_s2_persons_overlay.png'), dpi=200)
        plt.close()

        # RT overlay
        if has_rt_any:
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
