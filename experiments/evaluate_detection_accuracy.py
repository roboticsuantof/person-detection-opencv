import pandas as pd

METRICS_FILE = "results/metrics.csv"
GROUND_TRUTH_FILE = "results/ground_truth.csv"

def main():
    try:
        df_metrics = pd.read_csv(METRICS_FILE)
    except FileNotFoundError:
        print(f"No se encontró {METRICS_FILE}")
        return

    try:
        df_gt = pd.read_csv(GROUND_TRUTH_FILE)
    except FileNotFoundError:
        print(f"No se encontró {GROUND_TRUTH_FILE}")
        return

    # Nos quedamos solo con frame y persons
    if "frame" not in df_metrics.columns or "persons" not in df_metrics.columns:
        print("metrics.csv no contiene columnas 'frame' y 'persons'.")
        return

    df_metrics_simple = df_metrics[["frame", "persons"]]

    # Unir por frame
    df = pd.merge(df_gt, df_metrics_simple, on="frame", how="inner")

    if df.empty:
        print("No hay coincidencias de frames entre ground_truth.csv y metrics.csv.")
        return

    # Acierto si el conteo de personas coincide exactamente
    df["hit"] = (df["gt_persons"] == df["persons"])

    total = len(df)
    aciertos = df["hit"].sum()
    errores = total - aciertos
    precision = aciertos / total

    print("=== EVALUACIÓN DE PRECISIÓN DE DETECCIÓN ===")
    print(f"Total de casos evaluados      : {total}")
    print(f"Número de aciertos            : {aciertos}")
    print(f"Número de errores             : {errores}")
    print(f"Precisión de detección        : {precision * 100:.2f}%")

if __name__ == "__main__":
    main()
