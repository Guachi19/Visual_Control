from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

RUTA_MODELO = Path(__file__).parent / "models" / "hand_landmarker.task"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

if not RUTA_MODELO.is_file():
    raise FileNotFoundError(
        f"No se encontro el modelo de manos: {RUTA_MODELO}\n"
        f"Descargalo desde: {URL_MODELO}"
    )

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la camara")

opciones = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(RUTA_MODELO)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

try:
    with vision.HandLandmarker.create_from_options(opciones) as detector:
        while True:
            ok, imagen = camara.read()
            if not ok:
                print("No se pudo leer una imagen de la camara")
                break

            imagen = cv2.flip(imagen, 1)
            alto, ancho = imagen.shape[:2]
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            entrada = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
            tiempo_ms = time.monotonic_ns() // 1_000_000
            resultados = detector.detect_for_video(entrada, tiempo_ms)

            if resultados.hand_landmarks:
                for puntos_mano in resultados.hand_landmarks:
                    for conexion in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        inicio = puntos_mano[conexion.start]
                        fin = puntos_mano[conexion.end]
                        inicio_px = (int(inicio.x * ancho), int(inicio.y * alto))
                        fin_px = (int(fin.x * ancho), int(fin.y * alto))
                        cv2.line(imagen, inicio_px, fin_px, (0, 255, 0), 2)

                    for punto in puntos_mano:
                        posicion = (int(punto.x * ancho), int(punto.y * alto))
                        cv2.circle(imagen, posicion, 5, (0, 0, 255), cv2.FILLED)

                texto = "Mano detectada"
            else:
                texto = "No se detecta una mano"

            cv2.putText(
                imagen,
                texto,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Detector de mano - presiona q para salir", imagen)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camara.release()
    cv2.destroyAllWindows()
