"""Ejemplo para mostrar hasta dos manos y su lateralidad."""

import cv2
from mediapipe.tasks.python import vision

from detector_comun import ejecutar_detector, obtener_lado_espejado


def procesar_imagen(imagen, resultados):
    """Dibuja landmarks, lado y cantidad de manos detectadas.

    Args:
        imagen: Frame BGR sobre el que se dibuja.
        resultados: Resultado de MediaPipe, incluyendo landmarks y lados.
    """
    alto, ancho = imagen.shape[:2]

    for indice, puntos_mano in enumerate(resultados.hand_landmarks):
        categoria = resultados.handedness[indice][0]
        lado = obtener_lado_espejado(categoria.category_name)
        muneca = puntos_mano[0]
        posicion_muneca = (int(muneca.x * ancho), int(muneca.y * alto))
        color = (0, 255, 0) if lado == "Izq" else (255, 128, 0)

        cv2.putText(
            imagen,
            f"{lado} ({categoria.score:.0%})",
            (posicion_muneca[0] + 10, posicion_muneca[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        for conexion in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            inicio = puntos_mano[conexion.start]
            fin = puntos_mano[conexion.end]
            inicio_px = (int(inicio.x * ancho), int(inicio.y * alto))
            fin_px = (int(fin.x * ancho), int(fin.y * alto))
            cv2.line(imagen, inicio_px, fin_px, color, 2)

        for punto in puntos_mano:
            posicion = (int(punto.x * ancho), int(punto.y * alto))
            cv2.circle(imagen, posicion, 5, (0, 0, 255), cv2.FILLED)

    cv2.putText(
        imagen,
        f"Manos detectadas: {len(resultados.hand_landmarks)}/2",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


if __name__ == "__main__":
    ejecutar_detector(
        procesar_imagen,
        "Detector de dos manos - presiona q para salir",
    )
