"""Captura ejemplos etiquetados de la mano derecha para el alfabeto LSC."""

import csv
from collections import Counter
from pathlib import Path

import cv2
from mediapipe.tasks.python import vision

from detector_comun import ejecutar_detector, obtener_lado_espejado
from lsc_clasificador import normalizar_landmarks

RUTA_DATOS = Path(__file__).parent / "datos_lsc.csv"
COLUMNAS = ["etiqueta", "confianza"]
COLUMNAS.extend(
    f"{eje}{indice}"
    for indice in range(21)
    for eje in ("x", "y", "z")
)


class CapturadorLSC:
    """Mantiene etiqueta, conteos y landmarks durante una sesión de captura."""

    def __init__(self, writer, archivo):
        """Prepara el capturador y carga los conteos guardados previamente.

        Args:
            writer: Escritor CSV que recibe cada nuevo ejemplo.
            archivo: Archivo CSV abierto en modo anexar, que se vacía tras
                guardar cada fila para conservar los datos ante un cierre.
        """
        self.writer = writer
        self.archivo = archivo
        self.etiqueta = ""
        self.editando_etiqueta = False
        self.texto_etiqueta = ""
        self.landmarks_actuales = None
        self.confianza_actual = 0.0
        self.conteos = Counter()
        self.estado = "Enter: etiqueta | C: capturar | Q: salir"
        self.cargar_conteos_existentes()

    def cargar_conteos_existentes(self):
        """Lee cuántos ejemplos hay por etiqueta y valida el encabezado."""
        if not RUTA_DATOS.is_file() or RUTA_DATOS.stat().st_size == 0:
            return

        with RUTA_DATOS.open("r", newline="", encoding="utf-8") as entrada:
            lector = csv.DictReader(entrada)
            if lector.fieldnames != COLUMNAS:
                raise ValueError(
                    f"El formato de {RUTA_DATOS.name} no coincide con el "
                    "formato actual. Renombra o mueve ese archivo antes "
                    "de iniciar una nueva captura."
                )
            self.conteos.update(fila["etiqueta"] for fila in lector)

    def procesar_imagen(self, imagen, resultados):
        """Actualiza landmarks de la mano derecha y dibuja el estado actual.

        Args:
            imagen: Frame BGR sobre el que se dibuja la vista de captura.
            resultados: Landmarks y lateralidad detectados por MediaPipe.
        """
        alto, ancho = imagen.shape[:2]
        self.landmarks_actuales = None
        self.confianza_actual = 0.0
        mano_derecha_detectada = False

        for indice, puntos_mano in enumerate(resultados.hand_landmarks):
            categoria = resultados.handedness[indice][0]
            if obtener_lado_espejado(categoria.category_name) != "Der":
                continue

            mano_derecha_detectada = True
            self.landmarks_actuales = normalizar_landmarks(
                puntos_mano,
                ancho,
                alto,
            )
            self.confianza_actual = categoria.score

            for conexion in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                inicio = puntos_mano[conexion.start]
                fin = puntos_mano[conexion.end]
                cv2.line(
                    imagen,
                    (int(inicio.x * ancho), int(inicio.y * alto)),
                    (int(fin.x * ancho), int(fin.y * alto)),
                    (0, 255, 0),
                    2,
                )

            for punto in puntos_mano:
                cv2.circle(
                    imagen,
                    (int(punto.x * ancho), int(punto.y * alto)),
                    5,
                    (0, 0, 255),
                    cv2.FILLED,
                )
            break

        texto_mano = (
            "Mano derecha detectada"
            if mano_derecha_detectada
            else "No se detecta la mano derecha"
        )
        cv2.putText(
            imagen,
            texto_mano,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        texto_etiqueta = (
            f"Escribiendo etiqueta: {self.texto_etiqueta}_"
            if self.editando_etiqueta
            else f"Etiqueta: {self.etiqueta or '(sin seleccionar)'}"
        )
        cv2.putText(
            imagen,
            texto_etiqueta,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            imagen,
            f"Ejemplos guardados: {self.conteos[self.etiqueta]}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            imagen,
            self.estado[:70],
            (20, alto - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    def procesar_tecla(self, tecla):
        """Edita la etiqueta o guarda el ejemplo según la tecla recibida.

        Args:
            tecla: Código de tecla de OpenCV; Enter edita/acepta etiqueta,
                ``C`` captura y ``Esc`` cancela la edición.

        Returns:
            ``True`` si la tecla fue consumida por el capturador.
        """
        if self.editando_etiqueta:
            if tecla in (10, 13):
                etiqueta = self.texto_etiqueta.strip().lower()
                if etiqueta:
                    self.etiqueta = etiqueta
                    self.estado = f"Etiqueta seleccionada: {etiqueta}"
                else:
                    self.estado = "Etiqueta vacia; escribe una y pulsa Enter"
                self.texto_etiqueta = ""
                self.editando_etiqueta = False
                return True
            if tecla == 27:
                self.texto_etiqueta = ""
                self.editando_etiqueta = False
                self.estado = "Edicion de etiqueta cancelada"
                return True
            if tecla in (8, 127):
                self.texto_etiqueta = self.texto_etiqueta[:-1]
                return True
            if tecla in (ord("["), 241) and len(self.texto_etiqueta) < 30:
                self.texto_etiqueta += "ñ"
                return True
            if 32 <= tecla <= 126 and len(self.texto_etiqueta) < 30:
                caracter = chr(tecla)
                if caracter.isalnum() or caracter in " _-":
                    self.texto_etiqueta += caracter
                return True
            return True

        if tecla in (10, 13):
            self.editando_etiqueta = True
            self.texto_etiqueta = ""
            self.estado = "Escribe etiqueta; [ = ñ; Enter guarda; Esc cancela"
            return True

        if tecla == ord("c"):
            if not self.etiqueta:
                self.estado = "Primero define una etiqueta con Enter"
            elif self.landmarks_actuales is None:
                self.estado = "No hay mano derecha visible para capturar"
            else:
                self.writer.writerow(
                    [
                        self.etiqueta,
                        f"{self.confianza_actual:.6f}",
                        *(
                            f"{valor:.8f}"
                            for valor in self.landmarks_actuales
                        ),
                    ]
                )
                self.archivo.flush()
                self.conteos[self.etiqueta] += 1
                self.estado = f"Ejemplo guardado para {self.etiqueta}"
            return True

        return False


def iniciar_captura():
    """Valida o crea el CSV y abre la cámara para capturar ejemplos LSC.

    Los landmarks se guardan localmente en ``datos_lsc.csv``; no se guardan
    fotogramas ni imágenes.
    """
    archivo_existe = RUTA_DATOS.exists() and RUTA_DATOS.stat().st_size > 0
    if archivo_existe:
        with RUTA_DATOS.open("r", newline="", encoding="utf-8") as entrada:
            lector = csv.reader(entrada)
            if next(lector, []) != COLUMNAS:
                raise ValueError(
                    f"El formato de {RUTA_DATOS.name} no coincide con el "
                    "capturador. Renombra o mueve ese archivo para iniciar "
                    "una captura nueva."
                )

    with RUTA_DATOS.open("a", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        if not archivo_existe:
            writer.writerow(COLUMNAS)
            archivo.flush()

        capturador = CapturadorLSC(writer, archivo)
        ejecutar_detector(
            capturador.procesar_imagen,
            "Captura local de landmarks LSC",
            num_manos=2,
            procesar_tecla=capturador.procesar_tecla,
        )


if __name__ == "__main__":
    iniciar_captura()
