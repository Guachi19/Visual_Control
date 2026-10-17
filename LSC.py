"""Visualiza y clasifica las señas estáticas de la mano derecha en LSC."""

import cv2
from mediapipe.tasks.python import vision

from detector_comun import ejecutar_detector, obtener_lado_espejado
from lsc_clasificador import ClasificadorLSC, normalizar_landmarks


def crear_procesador(clasificador):
    """Crea el callback por frame que dibuja y clasifica la mano derecha.

    Args:
        clasificador: Instancia de :class:`ClasificadorLSC` ya cargada.

    Returns:
        Función ``(imagen, resultados)`` compatible con ``ejecutar_detector``.
    """
    def procesar_imagen(imagen, resultados):
        """Dibuja landmarks y la predicción de la mano derecha detectada.

        Args:
            imagen: Frame BGR sobre el que se dibuja.
            resultados: Resultado de MediaPipe con landmarks y lateralidad.
        """
        alto, ancho = imagen.shape[:2]
        mano_derecha_detectada = False

        for indice, puntos_mano in enumerate(resultados.hand_landmarks):
            categoria = resultados.handedness[indice][0]
            if obtener_lado_espejado(categoria.category_name) != "Der":
                continue

            mano_derecha_detectada = True
            color = (0, 255, 0)
            muneca = puntos_mano[0]
            posicion_muneca = (int(muneca.x * ancho), int(muneca.y * alto))

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
                f"Mano derecha ({categoria.score:.0%})",
                (posicion_muneca[0] + 10, posicion_muneca[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            caracteristicas = normalizar_landmarks(puntos_mano, ancho, alto)
            prediccion = clasificador.predecir(caracteristicas)
            cv2.putText(
                imagen,
                f"Prediccion: {prediccion.upper()}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
            )
            break

        if not mano_derecha_detectada:
            cv2.putText(
                imagen,
                "No se detecta la mano derecha",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    return procesar_imagen


if __name__ == "__main__":
    clasificador = ClasificadorLSC.cargar()
    ejecutar_detector(
        crear_procesador(clasificador),
        "LSC - mano derecha - presiona q para salir",
        num_manos=2,
    )
