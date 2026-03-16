import os
import pandas as pd

METRICS_FILE = "results/metrics.csv"
LANDMARKS_FILE = "results/landmarks.csv"

def human_size(bytes_size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def main():
    if not os.path.exists(METRICS_FILE):
        print(f"No se encontró {METRICS_FILE}")
        return

    df = pd.read_csv(METRICS_FILE)

    total_frames = len(df)
    frames_con_personas = (df["persons"] > 0).sum()
    frames_con_pose_valida = ((df["persons"] > 0) & (df["avg_landmarks"] > 0)).sum()

    print("=== RESUMEN DE REGISTRO ===")
    print(f"Total de frames registrados              : {total_frames}")
    print(f"Frames con al menos una persona          : {frames_con_personas}")
    print(f"Frames con pose válida                   : {frames_con_pose_valida}")

    if total_frames > 0:
        print(f"% frames con personas                    : {frames_con_personas / total_frames * 100:.2f}%")
    if frames_con_personas > 0:
        print(f"% frames con pose válida (entre con personas): {frames_con_pose_valida / frames_con_personas * 100:.2f}%")

    # Métricas de tasa de muestreo y tiempo de respuesta
    if "sampling_rate_hz" in df.columns:
        print("\n=== TASA DE MUESTREO (FPS) ===")
        print(f"FPS medio                                : {df['sampling_rate_hz'].mean():.2f}")
        print(f"FPS mínimo                               : {df['sampling_rate_hz'].min():.2f}")
        print(f"FPS máximo                               : {df['sampling_rate_hz'].max():.2f}")

    if "response_time_ms" in df.columns:
        print("\n=== TIEMPO DE RESPUESTA (ms) ===")
        print(f"Tiempo de respuesta medio                : {df['response_time_ms'].mean():.2f} ms")
        print(f"Tiempo de respuesta mínimo               : {df['response_time_ms'].min():.2f} ms")
        print(f"Tiempo de respuesta máximo               : {df['response_time_ms'].max():.2f} ms")

    # Tamaños de archivo
    print("\n=== TAMAÑO DE DOCUMENTOS DE REGISTRO ===")
    metrics_size = os.path.getsize(METRICS_FILE)
    print(f"metrics.csv   : {human_size(metrics_size)}")

    if os.path.exists(LANDMARKS_FILE):
        landmarks_size = os.path.getsize(LANDMARKS_FILE)
        print(f"landmarks.csv : {human_size(landmarks_size)}")
    else:
        print("landmarks.csv : no existe o no se ha generado aún.")

if __name__ == "__main__":
    main()
