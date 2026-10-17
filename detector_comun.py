"""Captura de cámara y configuración compartida del detector de manos."""

from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

RUTA_MODELO = Path(__file__).parent / "models" / "hand_landmarker.task"


def obtener_lado_espejado(nombre_categoria):
    """Convierte la lateralidad de MediaPipe en el lado visible tras el espejo.

    Args:
        nombre_categoria: Etiqueta de MediaPipe, normalmente ``Left`` o
            ``Right``.

    Returns:
        ``"Der"`` o ``"Izq"`` según el lado que aparece en la imagen espejada;
        devuelve la categoría original si no es una de las dos conocidas.
    """
    return {"Left": "Der", "Right": "Izq"}.get(
        nombre_categoria,
        nombre_categoria,
    )


def ejecutar_camara(
    procesar_imagen,
    titulo_ventana,
    procesar_tecla=None,
    debe_cerrar=None,
):
    """Abre la cámara y procesa imágenes hasta recibir la tecla de salida.

    Args:
        procesar_imagen: Función que recibe cada imagen BGR ya espejada y puede
            modificarla para dibujar resultados.
        titulo_ventana: Texto que identifica la ventana de OpenCV.
        procesar_tecla: Función opcional que recibe el código de una tecla y
            devuelve ``True`` si la consumió. La tecla ``q`` no consumida cierra
            el programa.
        debe_cerrar: Función opcional sin argumentos que devuelve ``True``
            cuando el procesador solicita cerrar el bucle de cámara.

    Raises:
        RuntimeError: Si no se puede abrir la cámara.
    """
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la camara")

    try:
        while True:
            ok, imagen = camara.read()
            if not ok:
                print("No se pudo leer una imagen de la camara")
                break

            imagen = cv2.flip(imagen, 1)
            procesar_imagen(imagen)
            cv2.imshow(titulo_ventana, imagen)

            tecla = cv2.waitKey(1) & 0xFF
            tecla_consumida = procesar_tecla(tecla) if procesar_tecla else False
            cierre_solicitado = debe_cerrar() if debe_cerrar else False
            if cierre_solicitado or (tecla == ord("q") and not tecla_consumida):
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


def ejecutar_detector(
    procesar_imagen,
    titulo_ventana,
    num_manos=2,
    procesar_tecla=None,
    debe_cerrar=None,
):
    """Ejecuta MediaPipe en cada imagen y entrega resultados al procesador.

    Args:
        procesar_imagen: Callback con firma ``(imagen, resultados)`` para
            procesar cada frame y sus landmarks.
        titulo_ventana: Título de la ventana de cámara.
        num_manos: Máximo de manos que MediaPipe buscará simultáneamente.
        procesar_tecla: Callback opcional de teclado, reenviado a
            :func:`ejecutar_camara`.
        debe_cerrar: Callback opcional de cierre, reenviado a
            :func:`ejecutar_camara`.

    Raises:
        FileNotFoundError: Si falta el modelo local ``hand_landmarker.task``.
    """
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            f"No se encontro el modelo de manos: {RUTA_MODELO}\n"
            "Ejecuta detectar_mano.py despues de descargar el modelo."
        )

    opciones = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(RUTA_MODELO)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_manos,
    )

    with vision.HandLandmarker.create_from_options(opciones) as detector:
        def procesar_con_detector(imagen):
            """Calcula landmarks del frame y los entrega al callback."""
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            entrada = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=imagen_rgb,
            )
            tiempo_ms = time.monotonic_ns() // 1_000_000
            resultados = detector.detect_for_video(entrada, tiempo_ms)
            procesar_imagen(imagen, resultados)

        ejecutar_camara(
            procesar_con_detector,
            titulo_ventana,
            procesar_tecla=procesar_tecla,
            debe_cerrar=debe_cerrar,
        )
