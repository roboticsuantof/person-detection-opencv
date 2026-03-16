#!/usr/bin/env python3
"""
combine_s1_replicas_v2.py — Consolida réplicas del escenario S1.

Soporta GT:
- Constante:            --gt <int>                 (default: 1)
- Global (mismo archivo): --gt_global <path.csv>
- Por réplica:          --gt_name ground_truth.csv (default) buscado dentro de cada S1_Rx

Formato GT aceptado:
  - Por frame:    (frame, gt_persons)
  - Por segmentos:(frame_start, frame_end, gt_persons)  [frame_end incluido]

Salidas (en --root):
  - summary_s1_replicas.csv    (lectura de summary_s1.csv si existe)
  - summary_s1_recomputed.csv  (recompute desde metrics + GT)
  - summary_s1_aggregate.csv   (promedio y peor caso)
  - fig_s1_fps_overlay.png
  - fig_s1_persons_overlay.png
  - fig_s1_rt_overlay.png (si existe)

Uso típico (GT por réplica):
  python combine_s1_replicas.py --root results/analysis_s1 --pattern "S1_R*" --skip_first 20 --gt_name ground_truth.csv
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
    df = pd.read_csv(gt_path)

    if {'frame', 'gt_persons'}.issubset(df.columns):
        gt = df[['frame', 'gt_persons']].copy()

    elif {'frame_start', 'frame_end', 'gt_persons'}.issubset(df.columns):
        rows = []
        for _, r in df.iterrows():
            a = int(r['frame_start']); b = int(r['frame_end']); v = int(r['gt_persons'])
            if b < a:
                continue
            for f in range(a, b + 1):
                rows.append((f, v))
        gt = pd.DataFrame(rows, columns=['frame', 'gt_persons'])
    else:
        raise ValueError(
            f'GT inválido: {gt_path}. Debe tener (frame, gt_persons) o (frame_start, frame_end, gt_persons). '
            f'Columnas: {list(df.columns)}'
        )

    gt = gt.sort_values('frame').drop_duplicates('frame', keep='last')
    gt['frame'] = gt['frame'].astype(int)
    gt['gt_persons'] = gt['gt_persons'].astype(int)
    return gt


def eval_s1(df_metrics: pd.DataFrame, gt_df: pd.DataFrame, skip_first: int):
    frame_col = pick_col(df_metrics, ['frame', 'frame_idx', 'frame_id'])
    persons_col = pick_col(df_metrics, ['persons', 'detected_persons', 'person_count'])
    fps_col = pick_col(df_metrics, ['fps', 'sampling_rate_hz'])
    rt_col = pick_col(df_metrics, ['response_time_ms', 'latency_ms', 'time_ms'])

    if frame_col is None or persons_col is None or fps_col is None:
        raise ValueError('metrics.csv debe tener frame/persons/fps')

    d = df_metrics.copy()
    if skip_first > 0:
        d = d[d[frame_col] >= int(skip_first)]

    N = int(len(d))
    persons = d[persons_col].astype(int).to_numpy()
    fps = d[fps_col].astype(float).to_numpy()

    rec = {
        'frames_used': N,
        'skip_first': int(skip_first),
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

    # merge con GT
    m = d[[frame_col, persons_col]].copy()
    m.columns = ['frame', 'persons']
    m['frame'] = m['frame'].astype(int)

    gt2 = gt_df.copy()
    if skip_first > 0:
        gt2 = gt2[gt2['frame'] >= int(skip_first)]

    dfe = pd.merge(gt2, m, on='frame', how='inner')
    if len(dfe) == 0:
        rec.update({'frames_eval_gt': 0, 'Acc_S1_exact': float('nan'), 'FNR_miss': float('nan'),
                    'FPR_extra': float('nan'), 'Overcount': float('nan')})
        return rec

    g = dfe['gt_persons'].astype(int).to_numpy()
    p = dfe['persons'].astype(int).to_numpy()

    rec.update({
        'frames_eval_gt': int(len(dfe)),
        'Acc_S1_exact': float(np.mean(p == g)),
        'FNR_miss': float(np.mean((g == 1) & (p == 0))),
        'FPR_extra': float(np.mean((g == 0) & (p > 0))),
        'Overcount': float(np.mean((g == 1) & (p > 1))),
    })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/analysis_s1')
    ap.add_argument('--pattern', default='S1_R*')
    ap.add_argument('--skip_first', type=int, default=20)

    ap.add_argument('--gt', type=int, default=1, help='GT constante si no hay archivos (default: 1)')
    ap.add_argument('--gt_global', default=None, help='Ruta a GT global (mismo para todas las réplicas)')
    ap.add_argument('--gt_name', default='ground_truth.csv', help='Nombre de GT por réplica dentro de cada carpeta')
    args = ap.parse_args()

    rep_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not rep_dirs:
        raise SystemExit(f'No se encontraron carpetas en {args.root} con patrón {args.pattern}')

    # Lectura de summary_s1.csv si existe
    summaries = []
    for rd in rep_dirs:
        sp = os.path.join(rd, 'summary_s1.csv')
        if os.path.exists(sp):
            s = pd.read_csv(sp)
            if len(s):
                s.insert(0, 'replica', os.path.basename(rd))
                summaries.append(s)
    if summaries:
        df_rep = pd.concat(summaries, ignore_index=True)
        df_rep.to_csv(os.path.join(args.root, 'summary_s1_replicas.csv'), index=False)

    # Preparar GT global o por réplica
    gt_global_df = None
    if args.gt_global:
        gt_global_df = load_gt(args.gt_global)

    recomputed = []
    valid_dirs = []
    any_rt = False

    for rd in rep_dirs:
        mp = os.path.join(rd, 'metrics.csv')
        if not os.path.exists(mp):
            continue

        dfm = pd.read_csv(mp)

        if gt_global_df is not None:
            gt_df = gt_global_df
            gt_src = os.path.basename(args.gt_global)
        else:
            gp = os.path.join(rd, args.gt_name)
            if os.path.exists(gp):
                gt_df = load_gt(gp)
                gt_src = args.gt_name
            else:
                # fallback: constante
                gt_df = pd.DataFrame({'frame': dfm[pick_col(dfm, ['frame','frame_idx','frame_id'])].astype(int),
                                      'gt_persons': int(args.gt)})
                gt_src = f'constant:{int(args.gt)}'

        rec = {'replica': os.path.basename(rd), 'gt_source': gt_src}
        # detect RT en metrics para overlay
        if 'response_time_ms' in dfm.columns:
            any_rt = True

        rec.update(eval_s1(dfm, gt_df, args.skip_first))
        recomputed.append(rec)
        valid_dirs.append(rd)

    if not recomputed:
        raise SystemExit('No se encontraron metrics.csv válidos para combinar.')

    df_re = pd.DataFrame(recomputed).sort_values('replica')
    df_re.to_csv(os.path.join(args.root, 'summary_s1_recomputed.csv'), index=False)

    # Agregado: promedio + peor caso
    agg = {
        'replicas_n': int(len(df_re)),
        'skip_first': int(args.skip_first),
        'Acc_mean': float(df_re['Acc_S1_exact'].mean()) if 'Acc_S1_exact' in df_re.columns else float('nan'),
        'Acc_worst': float(df_re['Acc_S1_exact'].min()) if 'Acc_S1_exact' in df_re.columns else float('nan'),
        'FNR_mean': float(df_re['FNR_miss'].mean()) if 'FNR_miss' in df_re.columns else float('nan'),
        'FNR_worst': float(df_re['FNR_miss'].max()) if 'FNR_miss' in df_re.columns else float('nan'),
        'FPR_mean': float(df_re['FPR_extra'].mean()) if 'FPR_extra' in df_re.columns else float('nan'),
        'FPR_worst': float(df_re['FPR_extra'].max()) if 'FPR_extra' in df_re.columns else float('nan'),
        'Overcount_mean': float(df_re['Overcount'].mean()) if 'Overcount' in df_re.columns else float('nan'),
        'Overcount_worst': float(df_re['Overcount'].max()) if 'Overcount' in df_re.columns else float('nan'),
        'fps_mean': float(df_re['fps_mean'].mean()),
        'fps_min_worst': float(df_re['fps_min'].min()),
    }

    if any_rt and 'response_time_ms_p95' in df_re.columns:
        agg.update({
            'rt_p95_mean': float(df_re['response_time_ms_p95'].mean()),
            'rt_p95_worst': float(df_re['response_time_ms_p95'].max()),
            'rt_max_worst': float(df_re['response_time_ms_max'].max()),
        })

    pd.DataFrame([agg]).to_csv(os.path.join(args.root, 'summary_s1_aggregate.csv'), index=False)

    # Overlays
    if len(valid_dirs) >= 2:
        # FPS overlay
        plt.figure(figsize=(10,4))
        for rd in valid_dirs:
            df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
            frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
            fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
            if args.skip_first > 0:
                df = df[df[frame_col] >= int(args.skip_first)]
            plt.plot(df[frame_col], df[fps_col].astype(float), label=os.path.basename(rd))
        plt.xlabel('Frame'); plt.ylabel('FPS')
        plt.title('S1: FPS por fotograma (superposición de réplicas)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, 'fig_s1_fps_overlay.png'), dpi=200)
        plt.close()

        # Persons overlay
        plt.figure(figsize=(10,4))
        for rd in valid_dirs:
            df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
            frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
            persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
            if args.skip_first > 0:
                df = df[df[frame_col] >= int(args.skip_first)]
            plt.plot(df[frame_col], df[persons_col].astype(float), label=os.path.basename(rd))
        plt.xlabel('Frame'); plt.ylabel('Personas')
        plt.title('S1: conteo de personas por fotograma (superposición de réplicas)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, 'fig_s1_persons_overlay.png'), dpi=200)
        plt.close()

        # RT overlay
        if any_rt:
            plt.figure(figsize=(10,4))
            for rd in valid_dirs:
                df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
                if 'response_time_ms' not in df.columns:
                    continue
                frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
                if args.skip_first > 0:
                    df = df[df[frame_col] >= int(args.skip_first)]
                plt.plot(df[frame_col], df['response_time_ms'].astype(float), label=os.path.basename(rd))
            plt.xlabel('Frame'); plt.ylabel('ms')
            plt.title('S1: tiempo de respuesta por fotograma (superposición de réplicas)')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.root, 'fig_s1_rt_overlay.png'), dpi=200)
            plt.close()

    print('OK ->', args.root)


if __name__ == '__main__':
    main()
