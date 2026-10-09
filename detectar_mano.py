import cv2
from mediapipe.tasks.python import vision

from detector_comun import ejecutar_detector


def procesar_imagen(imagen, resultados):
    alto, ancho = imagen.shape[:2]
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

    texto = "Mano detectada" if resultados.hand_landmarks else "No se detecta una mano"
    cv2.putText(
        imagen,
        texto,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


if __name__ == "__main__":
    ejecutar_detector(
        procesar_imagen,
        "Detector de mano - presiona q para salir",
        num_manos=1,
    )
