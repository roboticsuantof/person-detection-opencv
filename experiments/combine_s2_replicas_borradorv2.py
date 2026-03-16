#!/usr/bin/env python3
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
    return float(np.percentile(x, q)) if x.size else float("nan")

def load_gt(gt_path: str) -> pd.DataFrame:
    df = pd.read_csv(gt_path)

    # GT por frame
    if {"frame", "gt_persons"}.issubset(df.columns):
        gt = df[["frame", "gt_persons"]].copy()

    # GT por segmentos
    elif {"frame_start", "frame_end", "gt_persons"}.issubset(df.columns):
        rows = []
        for _, r in df.iterrows():
            a = int(r["frame_start"])
            b = int(r["frame_end"])
            v = int(r["gt_persons"])
            for f in range(a, b + 1):  # inclusive
                rows.append((f, v))
        gt = pd.DataFrame(rows, columns=["frame", "gt_persons"])

    else:
        raise ValueError(f"Formato GT inválido en {gt_path}. Columnas: {list(df.columns)}")

    gt = gt.sort_values("frame").drop_duplicates("frame", keep="last")
    return gt

def eval_vs_gt(df_metrics: pd.DataFrame, gt: pd.DataFrame, skip_first: int):
    frame_col = pick_col(df_metrics, ["frame", "frame_idx", "frame_id"])
    persons_col = pick_col(df_metrics, ["persons", "detected_persons", "person_count"])
    if frame_col is None or persons_col is None:
        raise ValueError("metrics.csv no tiene columnas de frame/persons")

    dfm = df_metrics[[frame_col, persons_col]].copy()
    dfm.columns = ["frame", "persons"]

    if skip_first > 0:
        dfm = dfm[dfm["frame"] >= skip_first]
        gt = gt[gt["frame"] >= skip_first]

    dfe = pd.merge(gt, dfm, on="frame", how="inner")
    if dfe.empty:
        return dict(frames_eval_gt=0, frames_correct=0, frames_under=0, frames_over=0,
                    Acc_S2=np.nan, R_under=np.nan, R_over=np.nan, MAE_count=np.nan, RMSE_count=np.nan)

    c = dfe["gt_persons"].astype(int).to_numpy()
    chat = dfe["persons"].astype(int).to_numpy()
    diff = chat - c

    N = len(dfe)
    correct = int(np.sum(diff == 0))
    under = int(np.sum(diff < 0))
    over = int(np.sum(diff > 0))

    return dict(
        frames_eval_gt=int(N),
        frames_correct=correct,
        frames_under=under,
        frames_over=over,
        Acc_S2=float(correct / N),
        R_under=float(under / N),
        R_over=float(over / N),
        MAE_count=float(np.mean(np.abs(diff))),
        RMSE_count=float(np.sqrt(np.mean(diff.astype(float) ** 2))),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/analysis_s2")
    ap.add_argument("--pattern", default="S2_R*")
    ap.add_argument("--skip_first", type=int, default=20)
    ap.add_argument("--gt_name", default="ground_truth.csv")  # dentro de cada réplica
    args = ap.parse_args()

    rep_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not rep_dirs:
        raise SystemExit(f"No se encontraron carpetas con patrón {args.pattern} en {args.root}")

    rows = []
    reps_ok = []

    for rd in rep_dirs:
        mp = os.path.join(rd, "metrics.csv")
        gp = os.path.join(rd, args.gt_name)

        if not os.path.exists(mp):
            continue
        if not os.path.exists(gp):
            raise SystemExit(f"Falta GT por réplica: {gp}")

        df = pd.read_csv(mp)
        gt = load_gt(gp)

        frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
        persons_col = pick_col(df, ["persons", "detected_persons", "person_count"])
        fps_col = pick_col(df, ["fps", "sampling_rate_hz"])
        rt_col = pick_col(df, ["response_time_ms", "latency_ms", "time_ms"])

        if frame_col is None or persons_col is None or fps_col is None:
            raise SystemExit(f"metrics.csv incompleto en {rd}: columnas={list(df.columns)}")

        d = df.copy()
        if args.skip_first > 0:
            d = d[d[frame_col] >= args.skip_first]

        persons = d[persons_col].astype(float).to_numpy()
        fps = d[fps_col].astype(float).to_numpy()

        rec = {
            "replica": os.path.basename(rd),
            "frames_used": int(len(d)),
            "skip_first": int(args.skip_first),
            "fps_mean": float(np.mean(fps)) if len(fps) else np.nan,
            "fps_min": float(np.min(fps)) if len(fps) else np.nan,
            "fps_max": float(np.max(fps)) if len(fps) else np.nan,
            "persons_mean_postwarmup": float(np.mean(persons)) if len(persons) else np.nan,
            "persons_max_postwarmup": float(np.max(persons)) if len(persons) else np.nan,
            "gt_source": args.gt_name,
        }

        if rt_col is not None and rt_col in d.columns:
            rt = d[rt_col].astype(float).to_numpy()
            rec.update({
                "response_time_ms_mean": float(np.mean(rt)) if len(rt) else np.nan,
                "response_time_ms_p95": safe_percentile(rt, 95),
                "response_time_ms_max": float(np.max(rt)) if len(rt) else np.nan,
            })

        rec.update(eval_vs_gt(df, gt, args.skip_first))

        rows.append(rec)
        reps_ok.append(rd)

    df_re = pd.DataFrame(rows).sort_values("replica")
    df_re.to_csv(os.path.join(args.root, "summary_s2_recomputed.csv"), index=False)

    # Agregado (promedio y peor caso)
    agg = {
        "replicas_n": int(len(df_re)),
        "skip_first": int(args.skip_first),
        "Acc_mean": float(df_re["Acc_S2"].mean()),
        "Acc_worst": float(df_re["Acc_S2"].min()),
        "R_under_mean": float(df_re["R_under"].mean()),
        "R_under_worst": float(df_re["R_under"].max()),
        "R_over_mean": float(df_re["R_over"].mean()),
        "R_over_worst": float(df_re["R_over"].max()),
        "MAE_mean": float(df_re["MAE_count"].mean()),
        "MAE_worst": float(df_re["MAE_count"].max()),
        "fps_mean": float(df_re["fps_mean"].mean()),
        "fps_min_worst": float(df_re["fps_min"].min()),
    }
    if "response_time_ms_p95" in df_re.columns:
        agg.update({
            "rt_p95_mean": float(df_re["response_time_ms_p95"].mean()),
            "rt_p95_worst": float(df_re["response_time_ms_p95"].max()),
            "rt_max_worst": float(df_re["response_time_ms_max"].max()),
        })
    pd.DataFrame([agg]).to_csv(os.path.join(args.root, "summary_s2_aggregate.csv"), index=False)

    # Overlays (FPS y RT). Para GT por réplica, no se dibuja una sola curva GT en overlay.
    if len(reps_ok) >= 2:
        # FPS overlay
        plt.figure(figsize=(10,4))
        for rd in reps_ok:
            df = pd.read_csv(os.path.join(rd, "metrics.csv"))
            frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
            fps_col = pick_col(df, ["fps", "sampling_rate_hz"])
            if args.skip_first > 0:
                df = df[df[frame_col] >= args.skip_first]
            plt.plot(df[frame_col], df[fps_col].astype(float), label=os.path.basename(rd))
        plt.xlabel("Frame"); plt.ylabel("FPS")
        plt.title("S2: FPS por fotograma (superposición de réplicas)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.root, "fig_s2_fps_overlay.png"), dpi=200)
        plt.close()

        # RT overlay (si existe)
        has_rt = all(os.path.exists(os.path.join(rd, "metrics.csv")) for rd in reps_ok) and \
                 any("response_time_ms" in pd.read_csv(os.path.join(rd, "metrics.csv"), nrows=5).columns for rd in reps_ok)
        if has_rt:
            plt.figure(figsize=(10,4))
            for rd in reps_ok:
                df = pd.read_csv(os.path.join(rd, "metrics.csv"))
                frame_col = pick_col(df, ["frame", "frame_idx", "frame_id"])
                if "response_time_ms" not in df.columns:
                    continue
                if args.skip_first > 0:
                    df = df[df[frame_col] >= args.skip_first]
                plt.plot(df[frame_col], df["response_time_ms"].astype(float), label=os.path.basename(rd))
            plt.xlabel("Frame"); plt.ylabel("ms")
            plt.title("S2: tiempo de respuesta por fotograma (superposición de réplicas)")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.root, "fig_s2_rt_overlay.png"), dpi=200)
            plt.close()

    print("OK ->", args.root)

if __name__ == "__main__":
    main()