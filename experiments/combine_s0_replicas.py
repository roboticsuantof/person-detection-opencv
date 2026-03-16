#!/usr/bin/env python3
"""
combine_s0_replicas.py — Consolida réplicas del escenario S0 (prueba negativa, 0 personas),
con soporte de warm-up vía --skip_first.

Estructura esperada:
  results/analysis_s0/S0_R1/metrics.csv
  results/analysis_s0/S0_R2/metrics.csv
  results/analysis_s0/S0_R3/metrics.csv

Salidas (en --root):
  - summary_s0_recomputed.csv  (métricas por réplica)
  - summary_s0_aggregate.csv   (promedio y peor caso entre réplicas)
  - fig_s0_fps_overlay.png
  - fig_s0_persons_overlay.png
  - fig_s0_rt_overlay.png      (si existe response_time_ms)

Métrica principal:
  - FPR_S0 = (# frames post warm-up con persons > 0) / N

Uso:
  python combine_s0_replicas.py --root results/analysis_s0 --pattern "S0_R*" --skip_first 20
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/analysis_s0', help='Carpeta raíz con réplicas')
    ap.add_argument('--pattern', default='S0_R*', help='Patrón de carpetas réplica')
    ap.add_argument('--skip_first', type=int, default=0, help='Descartar primeros N frames (warm-up)')
    args = ap.parse_args()

    rep_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not rep_dirs:
        raise SystemExit(f'No se encontraron réplicas en {args.root} con patrón {args.pattern}')

    rows = []
    valid_dirs = []
    any_rt = False

    for rd in rep_dirs:
        mp = os.path.join(rd, 'metrics.csv')
        if not os.path.exists(mp):
            continue

        df = pd.read_csv(mp)

        frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
        persons_col = pick_col(df, ['persons', 'detected_persons', 'person_count'])
        fps_col = pick_col(df, ['fps', 'sampling_rate_hz'])
        rt_col = pick_col(df, ['response_time_ms', 'latency_ms', 'time_ms'])

        missing = [name for name, col in [('frame', frame_col), ('persons', persons_col), ('fps', fps_col)] if col is None]
        if missing:
            raise SystemExit(f'Faltan columnas {missing} en {mp}. Columnas: {list(df.columns)}')

        d = df.copy()
        if args.skip_first > 0:
            d = d[d[frame_col] >= int(args.skip_first)]

        N = int(len(d))
        persons = d[persons_col].astype(float).to_numpy()
        fps = d[fps_col].astype(float).to_numpy()

        frames_fp = int(np.sum(persons > 0)) if N else 0
        fpr = frames_fp / N if N else float('nan')

        rec = {
            'replica': os.path.basename(rd),
            'frames_used': N,
            'skip_first': int(args.skip_first),
            'frames_with_persons': frames_fp,
            'FPR_S0': float(fpr),
            'fps_mean': float(np.mean(fps)) if N else float('nan'),
            'fps_min': float(np.min(fps)) if N else float('nan'),
            'fps_max': float(np.max(fps)) if N else float('nan'),
        }

        if rt_col is not None and rt_col in d.columns:
            any_rt = True
            rt = d[rt_col].astype(float).to_numpy()
            rec.update({
                'response_time_ms_mean': float(np.mean(rt)) if N else float('nan'),
                'response_time_ms_p95': safe_percentile(rt, 95),
                'response_time_ms_max': float(np.max(rt)) if N else float('nan'),
            })

        rows.append(rec)
        valid_dirs.append(rd)

    if not rows:
        raise SystemExit('No hay metrics.csv válidos para combinar.')

    df_re = pd.DataFrame(rows).sort_values('replica')
    df_re.to_csv(os.path.join(args.root, 'summary_s0_recomputed.csv'), index=False)

    # Agregado: promedio y peor caso
    agg = {
        'replicas_n': int(len(df_re)),
        'skip_first': int(args.skip_first),
        'FPR_mean': float(df_re['FPR_S0'].mean()),
        'FPR_worst': float(df_re['FPR_S0'].max()),
        'fps_mean': float(df_re['fps_mean'].mean()),
        'fps_min_worst': float(df_re['fps_min'].min()),
    }

    if any_rt and 'response_time_ms_p95' in df_re.columns:
        agg.update({
            'rt_p95_mean': float(df_re['response_time_ms_p95'].mean()),
            'rt_p95_worst': float(df_re['response_time_ms_p95'].max()),
            'rt_max_worst': float(df_re['response_time_ms_max'].max()),
        })

    pd.DataFrame([agg]).to_csv(os.path.join(args.root, 'summary_s0_aggregate.csv'), index=False)

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
        plt.title('S0: FPS por fotograma (superposición de réplicas)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, 'fig_s0_fps_overlay.png'), dpi=200)
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
        plt.title('S0: personas detectadas por fotograma (superposición de réplicas)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, 'fig_s0_persons_overlay.png'), dpi=200)
        plt.close()

        # RT overlay
        if any_rt:
            plt.figure(figsize=(10,4))
            for rd in valid_dirs:
                df = pd.read_csv(os.path.join(rd, 'metrics.csv'))
                frame_col = pick_col(df, ['frame', 'frame_idx', 'frame_id'])
                if 'response_time_ms' not in df.columns:
                    continue
                if args.skip_first > 0:
                    df = df[df[frame_col] >= int(args.skip_first)]
                plt.plot(df[frame_col], df['response_time_ms'].astype(float), label=os.path.basename(rd))
            plt.xlabel('Frame'); plt.ylabel('ms')
            plt.title('S0: tiempo de respuesta por fotograma (superposición de réplicas)')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.root, 'fig_s0_rt_overlay.png'), dpi=200)
            plt.close()

    print('OK ->', args.root)


if __name__ == '__main__':
    main()
