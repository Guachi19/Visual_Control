import ctypes
from ctypes import wintypes
import math

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from detector_comun import ejecutar_detector, obtener_lado_espejado

RANGO_PELLIZCO_PALMA = 0.15
RANGO_MAXIMO_PALMA = 1.5
UMBRAL_CAMBIO_GESTO = 0.30
UMBRAL_LIBERAR_CAMBIO = 0.35
MAX_DEDOS_EXTENDIDOS_MANO_CERRADA = 1
MODOS = ("Porcentual", "OpenClose", "Flechas", "Mouse")
MODO_ACTIVO = MODOS[0]
SELECTOR_ACTIVO = False
CLIC_IZQUIERDO_ACTIVO = False
CLIC_DERECHO_ACTIVO = False
FLECHA_VISIBLE = None
FLECHAS_ARMADA = False

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = ctypes.c_bool


class EntradaMouse(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class UnionEntrada(ctypes.Union):
    _fields_ = (("mi", EntradaMouse),)


class Entrada(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", UnionEntrada))


user32.SendInput.argtypes = [
    wintypes.UINT,
    ctypes.POINTER(Entrada),
    ctypes.c_int,
]
user32.SendInput.restype = wintypes.UINT

ANCHO_PANTALLA = user32.GetSystemMetrics(0)
ALTO_PANTALLA = user32.GetSystemMetrics(1)
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
UMBRAL_CLIC = 0.30
UMBRAL_LIBERAR_CLIC = 0.40


if ANCHO_PANTALLA <= 0 or ALTO_PANTALLA <= 0:
    raise RuntimeError("No se pudieron obtener las dimensiones de la pantalla")


def mover_cursor(punta_indice):
    x_normalizado = max(0.0, min(1.0, punta_indice.x))
    y_normalizado = max(0.0, min(1.0, punta_indice.y))
    x = round(x_normalizado * (ANCHO_PANTALLA - 1))
    y = round(y_normalizado * (ALTO_PANTALLA - 1))
    if not user32.SetCursorPos(x, y):
        raise ctypes.WinError(ctypes.get_last_error())


def hacer_clic(flags):
    entradas = (Entrada * len(flags))()
    for entrada, flag in zip(entradas, flags):
        entrada.type = 0
        entrada.union.mi.dwFlags = flag
    enviados = user32.SendInput(
        len(entradas),
        entradas,
        ctypes.sizeof(Entrada),
    )
    if enviados != len(entradas):
        raise ctypes.WinError(ctypes.get_last_error())


def distancia_entre(punto_a, punto_b, ancho, alto):
    return math.hypot(
        (punto_a.x - punto_b.x) * ancho,
        (punto_a.y - punto_b.y) * alto,
    )


def calcular_porcentaje_pellizco(puntos_mano, ancho, alto):
    distancia_pellizco = distancia_entre(
        puntos_mano[4], puntos_mano[8], ancho, alto
    )
    ancho_palma = distancia_entre(puntos_mano[5], puntos_mano[17], ancho, alto)
    escala = distancia_pellizco / max(ancho_palma, 1)
    porcentaje = max(
        0,
        min(
            100,
            (escala - RANGO_PELLIZCO_PALMA)
            / (RANGO_MAXIMO_PALMA - RANGO_PELLIZCO_PALMA)
            * 100,
        ),
    )
    return int(math.ceil(porcentaje / 5) * 5)


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
    global MODO_ACTIVO, SELECTOR_ACTIVO
    global CLIC_IZQUIERDO_ACTIVO, CLIC_DERECHO_ACTIVO
    global FLECHA_VISIBLE, FLECHAS_ARMADA

    alto, ancho = imagen.shape[:2]
    mano_izquierda_detectada = False
    selector_detectado = False
    zona_flechas = None
    ancho_zona = ancho / 3

    for indice, puntos_mano in enumerate(resultados.hand_landmarks):
        categoria = resultados.handedness[indice][0]
        if obtener_lado_espejado(categoria.category_name) != "Izq":
            continue

        mano_izquierda_detectada = True
        color = (0, 255, 0)
        distancia_selector = distancia_entre(
            puntos_mano[4], puntos_mano[20], ancho, alto
        )
        ancho_palma = distancia_entre(
            puntos_mano[5], puntos_mano[17], ancho, alto
        )
        escala_selector = distancia_selector / max(ancho_palma, 1)
        selector_detectado = escala_selector <= UMBRAL_CAMBIO_GESTO

        if selector_detectado and not SELECTOR_ACTIVO:
            modo_anterior = MODO_ACTIVO
            MODO_ACTIVO = MODOS[(MODOS.index(MODO_ACTIVO) + 1) % len(MODOS)]
            SELECTOR_ACTIVO = True
            if modo_anterior == "Flechas" or MODO_ACTIVO == "Flechas":
                FLECHAS_ARMADA = False
                FLECHA_VISIBLE = None
        elif escala_selector > UMBRAL_LIBERAR_CAMBIO:
            SELECTOR_ACTIVO = False

        contorno_palma = np.array(
            [
                (
                    int(puntos_mano[punto].x * ancho),
                    int(puntos_mano[punto].y * alto),
                )
                for punto in (0, 5, 9, 13, 17)
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

        if MODO_ACTIVO == "Mouse":
            punta_indice = puntos_mano[8]
            mover_cursor(punta_indice)
            cv2.circle(
                imagen,
                (int(punta_indice.x * ancho), int(punta_indice.y * alto)),
                10,
                (255, 0, 255),
                2,
            )
            distancia_medio = distancia_entre(
                puntos_mano[4], puntos_mano[12], ancho, alto
            ) / max(ancho_palma, 1)
            distancia_anular = distancia_entre(
                puntos_mano[4], puntos_mano[16], ancho, alto
            ) / max(ancho_palma, 1)

            clic_izquierdo = distancia_medio <= UMBRAL_CLIC
            clic_derecho = distancia_anular <= UMBRAL_CLIC

            if selector_detectado:
                CLIC_IZQUIERDO_ACTIVO = False
                CLIC_DERECHO_ACTIVO = False
            else:
                if clic_izquierdo and not CLIC_IZQUIERDO_ACTIVO:
                    hacer_clic((MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP))
                    CLIC_IZQUIERDO_ACTIVO = True
                elif distancia_medio > UMBRAL_LIBERAR_CLIC:
                    CLIC_IZQUIERDO_ACTIVO = False

                if clic_derecho and not CLIC_DERECHO_ACTIVO:
                    hacer_clic((MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP))
                    CLIC_DERECHO_ACTIVO = True
                elif distancia_anular > UMBRAL_LIBERAR_CLIC:
                    CLIC_DERECHO_ACTIVO = False

        if selector_detectado:
            cv2.putText(
                imagen,
                "Cambio de gesto",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        elif MODO_ACTIVO == "Porcentual":
            porcentaje = calcular_porcentaje_pellizco(puntos_mano, ancho, alto)
            cv2.putText(
                imagen,
                f"Porcentual: {porcentaje}%",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
        elif MODO_ACTIVO == "OpenClose":
            dedos = (
                ("Pulgar", None),
                ("Indice", (5, 6, 7)),
                ("Medio", (9, 10, 11)),
                ("Anular", (13, 14, 15)),
                ("Menique", (17, 18, 19)),
            )
            estados = []
            for nombre, articulaciones in dedos:
                if articulaciones is None:
                    punto_pulgar = (
                        puntos_mano[4].x * ancho,
                        puntos_mano[4].y * alto,
                    )
                    extendido = cv2.pointPolygonTest(
                        contorno_palma,
                        punto_pulgar,
                        False,
                    ) < 0
                else:
                    inicio, articulacion, fin = articulaciones
                    umbral = 145 if nombre == "Anular" else 160
                    extendido = (
                        angulo_articulacion(
                            puntos_mano,
                            inicio,
                            articulacion,
                            fin,
                            ancho,
                            alto,
                        )
                        >= umbral
                    )
                estados.append(extendido)

            cantidad_dedos_extendidos = sum(estados)
            if cantidad_dedos_extendidos == len(estados):
                estado_openclose = "1"
            elif (
                cantidad_dedos_extendidos
                <= MAX_DEDOS_EXTENDIDOS_MANO_CERRADA
            ):
                estado_openclose = "0"
            else:
                estado_openclose = "I"
            cv2.putText(
                imagen,
                estado_openclose,
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 0),
                3,
            )

        if MODO_ACTIVO == "Flechas":
            centro_palma_x = sum(
                puntos_mano[punto].x for punto in (0, 5, 9, 13, 17)
            ) / 5
            if centro_palma_x < 1 / 3:
                zona_flechas = "izquierda"
            elif centro_palma_x > 2 / 3:
                zona_flechas = "derecha"
            else:
                zona_flechas = "centro"

    if MODO_ACTIVO == "Flechas":
        if zona_flechas == "centro":
            FLECHAS_ARMADA = True
            FLECHA_VISIBLE = None
        elif zona_flechas in ("izquierda", "derecha") and FLECHAS_ARMADA:
            flecha = "<-" if zona_flechas == "izquierda" else "->"
            print(f"Flecha {flecha}")
            FLECHA_VISIBLE = flecha
            FLECHAS_ARMADA = False
        elif zona_flechas is None:
            FLECHAS_ARMADA = False
            FLECHA_VISIBLE = None
        cv2.line(imagen, (ancho // 3, 0), (ancho // 3, alto), (100, 100, 100), 1)
        cv2.line(
            imagen,
            (2 * ancho // 3, 0),
            (2 * ancho // 3, alto),
            (100, 100, 100),
            1,
        )
        for etiqueta, centro_x in (
            ("IZQ", ancho // 6),
            ("CENTRO", ancho // 2),
            ("DER", 5 * ancho // 6),
        ):
            cv2.putText(
                imagen,
                etiqueta,
                (centro_x - 35, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
            )
    else:
        FLECHAS_ARMADA = False
        FLECHA_VISIBLE = None

    texto = (
        "Mano izquierda detectada"
        if mano_izquierda_detectada
        else "No se detecta la mano izquierda"
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
    if FLECHA_VISIBLE is not None:
        cv2.putText(
            imagen,
            f"Flecha: {FLECHA_VISIBLE}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )


if __name__ == "__main__":
    ejecutar_detector(
        procesar_imagen,
        "Gestos - presiona q para salir",
    )
