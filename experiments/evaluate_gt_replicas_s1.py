import os
import glob
import pandas as pd
import numpy as np

GROUND_TRUTH_FILE = "results/ground_truth.csv"
ROOT = "results/analysis_s1"
PATTERN = "S1_R*"
SKIP_FIRST = 20  # warm-up

def eval_one(metrics_path: str, df_gt: pd.DataFrame) -> dict:
    df_m = pd.read_csv(metrics_path)

    if "frame" not in df_m.columns or "persons" not in df_m.columns:
        raise ValueError(f"{metrics_path} no tiene columnas 'frame' y 'persons'.")

    df_m = df_m[["frame", "persons"]].copy()

    # aplicar warm-up sobre metrics (y opcionalmente GT si también parte en 0)
    df_m = df_m[df_m["frame"] >= SKIP_FIRST]

    df = pd.merge(df_gt, df_m, on="frame", how="inner")
    if df.empty:
        return {
            "replica": os.path.basename(os.path.dirname(metrics_path)),
            "frames_eval": 0,
            "hits": 0,
            "errors": 0,
            "accuracy": np.nan,
        }

    df["hit"] = (df["gt_persons"] == df["persons"])
    total = len(df)
    hits = int(df["hit"].sum())
    errors = total - hits
    acc = hits / total if total > 0 else np.nan

    return {
        "replica": os.path.basename(os.path.dirname(metrics_path)),
        "frames_eval": total,
        "hits": hits,
        "errors": errors,
        "accuracy": acc,
    }

def main():
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"No se encontró {GROUND_TRUTH_FILE}")
        return

    df_gt = pd.read_csv(GROUND_TRUTH_FILE)
    if "frame" not in df_gt.columns or "gt_persons" not in df_gt.columns:
        print("ground_truth.csv debe contener columnas: 'frame' y 'gt_persons'.")
        return

    # Si tu GT incluye warm-up, puedes filtrarlo también:
    df_gt = df_gt[df_gt["frame"] >= SKIP_FIRST]

    metrics_files = sorted(glob.glob(os.path.join(ROOT, PATTERN, "metrics.csv")))
    if not metrics_files:
        print(f"No se encontraron metrics.csv en {ROOT}/{PATTERN}/")
        return

    rows = []
    for mp in metrics_files:
        rows.append(eval_one(mp, df_gt))

    df_out = pd.DataFrame(rows).sort_values("replica")
    print(df_out)

    # agregado (promedio y peor caso)
    acc_mean = df_out["accuracy"].mean()
    acc_worst = df_out["accuracy"].min()
    print("\n=== AGREGADO ===")
    print(f"Accuracy promedio: {acc_mean*100:.2f}%")
    print(f"Accuracy peor caso: {acc_worst*100:.2f}%")

    out_csv = os.path.join(ROOT, "accuracy_s1_replicas.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nGuardado: {out_csv}")

if __name__ == "__main__":
    main()