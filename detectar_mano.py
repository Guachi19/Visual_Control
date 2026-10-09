import cv2
import mediapipe as mp

manos_mp = mp.solutions.hands
dibujo_mp = mp.solutions.drawing_utils

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

try:
    with manos_mp.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        while True:
            ok, imagen = camara.read()
            if not ok:
                print("No se pudo leer una imagen de la cámara")
                break

            imagen = cv2.flip(imagen, 1)
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            resultados = detector.process(imagen_rgb)

            if resultados.multi_hand_landmarks:
                for puntos_mano in resultados.multi_hand_landmarks:
                    dibujo_mp.draw_landmarks(
                        imagen,
                        puntos_mano,
                        manos_mp.HAND_CONNECTIONS,
                    )
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
