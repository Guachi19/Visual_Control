import math

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from detector_comun import ejecutar_detector, obtener_lado_espejado


def angulo_articulacion(puntos_mano, inicio, articulacion, fin, ancho, alto):
    punto_a = (
        puntos_mano[inicio].x * ancho,
        puntos_mano[inicio].y * alto,
    )
    punto_b = (
        puntos_mano[articulacion].x * ancho,
        puntos_mano[articulacion].y * alto,
    )
    punto_c = (
        puntos_mano[fin].x * ancho,
        puntos_mano[fin].y * alto,
    )
    vector_ba = (punto_a[0] - punto_b[0], punto_a[1] - punto_b[1])
    vector_bc = (punto_c[0] - punto_b[0], punto_c[1] - punto_b[1])
    magnitud = math.hypot(*vector_ba) * math.hypot(*vector_bc)
    if magnitud == 0:
        return 0

    coseno = (
        vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
    ) / magnitud
    return math.degrees(math.acos(max(-1.0, min(1.0, coseno))))


def procesar_imagen(imagen, resultados):
    alto, ancho = imagen.shape[:2]
    mano_derecha_detectada = False
    contador_binario = None

    for indice, puntos_mano in enumerate(resultados.hand_landmarks):
        categoria = resultados.handedness[indice][0]
        lado = obtener_lado_espejado(categoria.category_name)
        if lado != "Der":
            continue

        mano_derecha_detectada = True
        color = (0, 255, 0)
        muneca = puntos_mano[0]
        posicion_muneca = (int(muneca.x * ancho), int(muneca.y * alto))
        contorno_palma = np.array(
            [
                (
                    int(puntos_mano[indice].x * ancho),
                    int(puntos_mano[indice].y * alto),
                )
                for indice in (0, 5, 9, 13, 17)
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            imagen,
            [contorno_palma],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2,
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

        dedos = (
            ("P", 4, None),
            ("I", 8, (5, 6, 7)),
            ("C", 12, (9, 10, 11)),
            ("A", 16, (13, 14, 15)),
            ("M", 20, (17, 18, 19)),
        )
        bits = []
        for nombre, punta, articulaciones in dedos:
            if articulaciones is None:
                punta_pulgar = (
                    puntos_mano[4].x * ancho,
                    puntos_mano[4].y * alto,
                )
                extendido = cv2.pointPolygonTest(
                    contorno_palma,
                    punta_pulgar,
                    False,
                ) < 0
            else:
                inicio, articulacion, fin = articulaciones
                umbral_extension = 145 if nombre == "A" else 160
                extendido = (
                    angulo_articulacion(
                        puntos_mano,
                        inicio,
                        articulacion,
                        fin,
                        ancho,
                        alto,
                    )
                    >= umbral_extension
                )
            bits.append(int(extendido))

            punto_punta = puntos_mano[punta]
            posicion_etiqueta = (
                int(punto_punta.x * ancho) + 8,
                int(punto_punta.y * alto) - 8,
            )
            cv2.putText(
                imagen,
                f"{nombre}:{int(extendido)}",
                posicion_etiqueta,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if extendido else (0, 0, 255),
                2,
            )

        secuencia_binaria = "".join(map(str, reversed(bits)))
        valor_decimal = int(secuencia_binaria, 2)
        contador_binario = (
            f"Binario (Me-A-C-I-P): {secuencia_binaria} = {valor_decimal}"
        )

        cv2.putText(
            imagen,
            f"Mano Derecha ({categoria.score:.0%})",
            (posicion_muneca[0] + 10, posicion_muneca[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    texto = (
        "Mano Derecha detectada"
        if mano_derecha_detectada
        else "No se detecta la mano Deercha"
    )
    cv2.putText(
        imagen,
        texto,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    if contador_binario is not None:
        cv2.putText(
            imagen,
            contador_binario,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )


if __name__ == "__main__":
    ejecutar_detector(
        procesar_imagen,
        "Detector de mano Derecha - presiona q para salir",
    )
