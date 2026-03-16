import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np

# -------------------- CONFIG --------------------
CAMERA_ID = 0
YOLO_MODEL = "yolov8s-pose.pt"   # modelo de pose recomendado
SAVE_RESULTS_FOLDER = "results/frames"
METRICS_FILE = "results/metrics.csv"
LANDMARKS_FILE = "results/landmarks.csv"

SAVE_METRICS = True
SAVE_LANDMARKS = True
SAVE_CROPS = True

os.makedirs("results", exist_ok=True)
os.makedirs(SAVE_RESULTS_FOLDER, exist_ok=True)

# ============= PARÁMETROS DE ROBUSTEZ (ANTI FALSOS POSITIVOS) =============

# Umbral de confianza de detección (YOLO). Sube esto si sigue detectando cosas raras.
DETECTION_CONF_THRESH = 0.80  # antes usábamos ~0.55

# Mínimo porcentaje de área de la imagen que debe ocupar la caja de una "persona"
MIN_BOX_REL_AREA = 0.01  # 1% del área total del frame

# Mínimo de puntos de pose visibles y su confianza mínima
MIN_VISIBLE_LANDMARKS = 6
MIN_LANDMARK_CONF = 0.5
MIN_AVG_LANDMARK_CONF = 0.6

# ======================= ESQUELETO COCO =======================
# Conexiones oficiales (17 keypoints COCO)
SKELETON_COCO = [
    (0, 1), (1, 3), (0, 2), (2, 4),        # ojos — orejas
    (5, 6),                                # hombros
    (5, 7), (7, 9),                        # brazo izquierdo
    (6, 8), (8, 10),                       # brazo derecho
    (11, 12),                              # caderas
    (5, 11), (6, 12),                      # hombro → cadera
    (11, 13), (13, 15),                    # pierna izquierda
    (12, 14), (14, 16)                     # pierna derecha
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

try:
    while True:
        # ------------------------------------------------------------------
        # Medición de tiempo de respuesta (entrada → salida del frame)
        # ------------------------------------------------------------------
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        # ---------------- INFERENCIA YOLOv8-POSE ----------------
        # conf global (primera pasada)
        results = model(frame, conf=DETECTION_CONF_THRESH, imgsz=640)[0]

        num_persons = 0
        all_landmarks = 0

        boxes = results.boxes
        keypoints = results.keypoints

        if boxes is not None and keypoints is not None:
            for i, pose in enumerate(keypoints):
                if pose is None:
                    continue

                box = boxes[i]

                # Confianza de detección de la caja
                det_conf = float(box.conf[0])
                if det_conf < DETECTION_CONF_THRESH:
                    continue  # descartamos detección débil

                # Solo clase persona (en modelos pose normalmente ya es persona=0)
                if box.cls is not None:
                    cls_id = int(box.cls[0])
                    # Si quisieras filtrar explícitamente, podrías dejar solo cls_id == 0

                # Bounding box y área relativa
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)

                box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
                if frame_area > 0:
                    rel_area = box_area / frame_area
                else:
                    rel_area = 0.0

                # Filtrar cajas demasiado pequeñas (objetos chicos que no son personas reales)
                if rel_area < MIN_BOX_REL_AREA:
                    continue

                # Keypoints (x,y) y confidencias
                kpts = pose.xy[0].cpu().numpy()        # (17 x 2)
                kconf = pose.conf[0].cpu().numpy()     # (17)

                # ---------------- FILTRO POR CALIDAD DE POSE ----------------
                visible_mask = kconf >= MIN_LANDMARK_CONF
                visible_count = int(np.count_nonzero(visible_mask))

                if visible_count < MIN_VISIBLE_LANDMARKS:
                    # Muy pocos puntos confiables -> probablemente falso positivo
                    continue

                avg_conf_visible = float(kconf[visible_mask].mean())
                if avg_conf_visible < MIN_AVG_LANDMARK_CONF:
                    # La mayoría de los puntos son débiles
                    continue

                # Si llega hasta aquí, consideramos que la detección es válida
                num_persons += 1
                all_landmarks += visible_count

                # ---------------- DIBUJAR ESQUELETO COCO ----------------
                for (j1, j2) in SKELETON_COCO:
                    if kconf[j1] > MIN_LANDMARK_CONF and kconf[j2] > MIN_LANDMARK_CONF:
                        x1_l, y1_l = int(kpts[j1][0]), int(kpts[j1][1])
                        x2_l, y2_l = int(kpts[j2][0]), int(kpts[j2][1])
                        cv2.line(frame, (x1_l, y1_l), (x2_l, y2_l), (0, 255, 255), 3)

                # ---------------- DIBUJAR KEYPOINTS ----------------
                for (x, y), conf in zip(kpts, kconf):
                    if conf > MIN_LANDMARK_CONF:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 128, 255), -1)

                # ---------------- RECORTE por persona (para guardar) ----------------
                if SAVE_CROPS:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(f"{SAVE_RESULTS_FOLDER}/crop_{frame_idx}_p{num_persons}.png", crop)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Guardar landmarks por persona
                if SAVE_LANDMARKS:
                    rows = []
                    for idx, ((x, y), conf) in enumerate(zip(kpts, kconf)):
                        rows.append([frame_idx, num_persons, idx, x, y, conf])

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

        # ---------------- FPS y TIEMPO DE RESPUESTA ----------------
        t_end = time.time()
        frame_time = t_end - t_start          # tiempo total por frame (s)
        response_time_ms = frame_time * 1000  # tiempo de respuesta (ms)

        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        fps_smooth = fps if fps_smooth is None else alpha * fps_smooth + (1 - alpha) * fps

        # ---------------- TEXTO EN PANTALLA ----------------
        cv2.putText(frame, f"Personas: {num_persons}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"T_muestreo: {fps_smooth:.1f} Hz", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"T_respuesta: {response_time_ms:.1f} ms", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)

        # Mostrar frame
        cv2.imshow("YOLOv8-Pose", frame)

        # ---------------- GUARDAR MÉTRICAS ----------------
        if SAVE_METRICS:
            avg_landmarks = all_landmarks / num_persons if num_persons > 0 else 0

            df_m = pd.DataFrame([{
                "timestamp": time.time(),
                "frame": frame_idx,
                "persons": num_persons,
                "fps": fps_smooth,                  # tasa efectiva en tiempo real
                "sampling_rate_hz": fps_smooth,     # alias: tasa de muestreo
                "response_time_ms": response_time_ms,
                "avg_landmarks": avg_landmarks
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
