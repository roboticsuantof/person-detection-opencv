import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np
from collections import deque

def draw_label(img, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX,
               scale=0.7, thickness=2, pad=6,
               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Dibuja texto con fondo sólido. (x, y) es la esquina superior izquierda del bloque.
    """
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x1, y1 = x, y
    x2, y2 = x + tw + 2 * pad, y + th + 2 * pad

    # Fondo (rectángulo relleno)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)

    # Texto (dentro del rectángulo)
    tx, ty = x + pad, y + pad + th
    cv2.putText(img, text, (tx, ty), font, scale, text_color, thickness, cv2.LINE_AA)

# -------------------- CONFIG --------------------
CAMERA_ID = 0
YOLO_MODEL = "yolov8s-pose.pt"   # modelo de pose
SAVE_RESULTS_FOLDER = "results/frames"
METRICS_FILE = "results/metrics.csv"
LANDMARKS_FILE = "results/landmarks.csv"

SAVE_METRICS = True
SAVE_LANDMARKS = True
SAVE_CROPS = True

os.makedirs("results", exist_ok=True)
os.makedirs(SAVE_RESULTS_FOLDER, exist_ok=True)

# ============= PARÁMETROS DE ROBUSTEZ (ANTI FALSOS POSITIVOS) =============
DETECTION_CONF_THRESH = 0.65
MIN_BOX_REL_AREA = 0.01

MIN_VISIBLE_LANDMARKS = 6
MIN_LANDMARK_CONF = 0.5
MIN_AVG_LANDMARK_CONF = 0.6

# ======================= HISTÉRESIS (ESTABILIZACIÓN DEL CONTEO) =======================
# Presencia (0 vs >=1) con persistencia
HYST_ENABLE = True
M_IN = 3     # frames consecutivos con presencia para "entrar"
M_OUT = 5    # frames consecutivos sin presencia para "salir"

# Suavizado del conteo cuando hay presencia:
# tomamos la mediana de los últimos W conteos >0 (reduce picos 2,3 por errores)
COUNT_WINDOW = 5

# ======================= ESQUELETO COCO =======================
SKELETON_COCO = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# -------------------- CARGAR YOLOv8-POSE --------------------
print("Cargando modelo YOLOv8-Pose…")
model = YOLO(YOLO_MODEL)

# -------------------- VIDEO --------------------
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

print("\nSistema funcionando — Presiona 'q' para salir.\n")

frame_idx = 0
fps_smooth = None
alpha = 0.9  # suavizado exponencial FPS

# ---------- Estado de histéresis ----------
presence_state = 0  # 0: no presencia estable, 1: presencia estable
in_count = 0
out_count = 0
count_hist = deque(maxlen=COUNT_WINDOW)  # historial de conteos crudos (>0 y también 0)

def apply_hysteresis(raw_count: int) -> int:
    """
    Retorna conteo estabilizado.
    - Primero estabiliza presencia (0 vs >=1) con M_IN/M_OUT.
    - Luego, si hay presencia estable, suaviza conteo usando mediana de ventana.
    """
    global presence_state, in_count, out_count, count_hist

    # actualizar historial
    count_hist.append(int(raw_count))

    # presencia cruda
    raw_presence = 1 if raw_count > 0 else 0

    if raw_presence == 1:
        in_count += 1
        out_count = 0
    else:
        out_count += 1
        in_count = 0

    if presence_state == 0 and in_count >= M_IN:
        presence_state = 1
    elif presence_state == 1 and out_count >= M_OUT:
        presence_state = 0

    if presence_state == 0:
        return 0

    # presencia estable: estimar conteo (mediana de los últimos valores > 0)
    nonzero = [c for c in count_hist if c > 0]
    if len(nonzero) == 0:
        return 1  # fallback razonable para S1
    return int(round(float(np.median(nonzero))))

try:
    while True:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        # ---------------- INFERENCIA YOLOv8-POSE ----------------
        results = model(frame, conf=DETECTION_CONF_THRESH, imgsz=640)[0]

        num_persons_raw = 0
        all_landmarks = 0

        boxes = results.boxes
        keypoints = results.keypoints

        if boxes is not None and keypoints is not None:
            for i, pose in enumerate(keypoints):
                if pose is None:
                    continue

                box = boxes[i]
                det_conf = float(box.conf[0])
                if det_conf < DETECTION_CONF_THRESH:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)

                box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
                rel_area = (box_area / frame_area) if frame_area > 0 else 0.0
                if rel_area < MIN_BOX_REL_AREA:
                    continue

                kpts = pose.xy[0].cpu().numpy()        # (17 x 2)
                kconf = pose.conf[0].cpu().numpy()     # (17)

                visible_mask = kconf >= MIN_LANDMARK_CONF
                visible_count = int(np.count_nonzero(visible_mask))
                if visible_count < MIN_VISIBLE_LANDMARKS:
                    continue

                avg_conf_visible = float(kconf[visible_mask].mean())
                if avg_conf_visible < MIN_AVG_LANDMARK_CONF:
                    continue

                # detección válida
                num_persons_raw += 1
                all_landmarks += visible_count

                # dibujar esqueleto
                for (j1, j2) in SKELETON_COCO:
                    if kconf[j1] > MIN_LANDMARK_CONF and kconf[j2] > MIN_LANDMARK_CONF:
                        x1_l, y1_l = int(kpts[j1][0]), int(kpts[j1][1])
                        x2_l, y2_l = int(kpts[j2][0]), int(kpts[j2][1])
                        cv2.line(frame, (x1_l, y1_l), (x2_l, y2_l), (0, 255, 255), 3)

                # dibujar keypoints
                for (x, y), conf in zip(kpts, kconf):
                    if conf > MIN_LANDMARK_CONF:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 128, 255), -1)

                # recorte (opcional)
                if SAVE_CROPS:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(f"{SAVE_RESULTS_FOLDER}/crop_{frame_idx}_p{num_persons_raw}.png", crop)

                # bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # guardar landmarks (por persona detectada)
                if SAVE_LANDMARKS:
                    rows = []
                    for idx, ((x, y), conf) in enumerate(zip(kpts, kconf)):
                        rows.append([frame_idx, num_persons_raw, idx, x, y, conf])

                    df_lm = pd.DataFrame(
                        rows,
                        columns=["frame", "person", "landmark_id", "x", "y", "conf"]
                    )
                    df_lm.to_csv(
                        LANDMARKS_FILE,
                        mode="a",
                        header=not os.path.exists(LANDMARKS_FILE),
                        index=False
                    )

        # ---------------- HISTÉRESIS: conteo estabilizado ----------------
        if HYST_ENABLE:
            num_persons = apply_hysteresis(num_persons_raw)
        else:
            num_persons = num_persons_raw

        # ---------------- FPS y TIEMPO DE RESPUESTA ----------------
        t_end = time.time()
        frame_time = t_end - t_start
        response_time_ms = frame_time * 1000

        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        fps_smooth = fps if fps_smooth is None else alpha * fps_smooth + (1 - alpha) * fps

        # ---------------- TEXTO EN PANTALLA ----------------
        x0, y0 = 15, 15
        dy = 34

        draw_label(frame, f"Personas (est.): {num_persons}", x0, y0, scale=0.8, thickness=2)
        draw_label(frame, f"Personas (raw): {num_persons_raw}", x0, y0 + dy, scale=0.8, thickness=2)
        draw_label(frame, f"FPS: {fps_smooth:.1f}", x0, y0 + 2*dy, scale=0.75, thickness=2)
        draw_label(frame, f"T_muestreo: {fps_smooth:.1f} Hz", x0, y0 + 3*dy, scale=0.75, thickness=2)
        draw_label(frame, f"T_respuesta: {response_time_ms:.1f} ms", x0, y0 + 4*dy, scale=0.75, thickness=2)

        if HYST_ENABLE:
            draw_label(frame, f"HYST: Min={M_IN} Mout={M_OUT} W={COUNT_WINDOW}",
                       x0, y0 + 5*dy, scale=0.6, thickness=2)

        cv2.imshow("YOLOv8-Pose", frame)

        # ---------------- GUARDAR MÉTRICAS ----------------
        if SAVE_METRICS:
            avg_landmarks = all_landmarks / num_persons_raw if num_persons_raw > 0 else 0

            df_m = pd.DataFrame([{
                "timestamp": time.time(),
                "frame": frame_idx,
                "persons": num_persons,                 # CONTEO ESTABILIZADO (recomendado para evaluación)
                "persons_raw": num_persons_raw,         # conteo crudo (para diagnóstico)
                "fps": fps_smooth,
                "sampling_rate_hz": fps_smooth,
                "response_time_ms": response_time_ms,
                "avg_landmarks": avg_landmarks,
                "hyst_enabled": int(HYST_ENABLE),
                "hyst_m_in": int(M_IN),
                "hyst_m_out": int(M_OUT),
                "hyst_window": int(COUNT_WINDOW),
            }])

            df_m.to_csv(
                METRICS_FILE,
                mode="a",
                header=not os.path.exists(METRICS_FILE),
                index=False
            )

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado.")