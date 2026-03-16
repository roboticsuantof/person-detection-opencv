import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

# Detectar columnas de frame y personas como antes
frame_candidates = ["frame", "frame_idx", "frame_index"]
frame_col = next((c for c in frame_candidates if c in df.columns), None)
if frame_col is None:
    raise KeyError(f"No se encontró columna de frame. Columnas: {list(df.columns)}")

persons_candidates = ["persons", "detected_persons", "num_persons"]
persons_col = next((c for c in persons_candidates if c in df.columns), None)
if persons_col is None:
    raise KeyError(f"No se encontró columna de personas. Columnas: {list(df.columns)}")

# ---------- FPS ----------
plt.figure()
plt.plot(df[frame_col], df["fps"], label="FPS")
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.title("FPS en el tiempo")
plt.grid()
plt.legend()
plt.show()

# ---------- PERSONAS ----------
plt.figure()
plt.plot(df[frame_col], df[persons_col], label="Personas detectadas", color="red")
plt.xlabel("Frame")
plt.ylabel("Cantidad de personas")
plt.title("Personas detectadas por frame")
plt.grid()
plt.legend()
plt.show()

# ---------- LANDMARKS ----------
if "avg_landmarks" in df.columns:
    plt.figure()
    plt.plot(df[frame_col], df["avg_landmarks"], label="Landmarks promedio", color="green")
    plt.xlabel("Frame")
    plt.ylabel("Landmarks promedio")
    plt.title("Landmarks detectados por frame")
    plt.grid()
    plt.legend()
    plt.show()

# ---------- TIEMPO DE RESPUESTA ----------
if "response_time_ms" in df.columns:
    plt.figure()
    plt.plot(df[frame_col], df["response_time_ms"], label="Tiempo de respuesta (ms)")
    plt.xlabel("Frame")
    plt.ylabel("ms")
    plt.title("Tiempo de respuesta por frame")
    plt.grid()
    plt.legend()
    plt.show()
