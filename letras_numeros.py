"""Programa coordinador para alternar entre reconocimiento LSC y números."""

import time

import cv2

from LSC import crear_procesador
from detector_comun import ejecutar_detector, obtener_lado_espejado
from lsc_clasificador import ClasificadorLSC, normalizar_landmarks
from numeros import procesar_imagen as procesar_numeros

DURACION_CIERRE_Q = 2.5


class ControlGestosManoDerecha:
    """Controla el cambio de modo por zonas y el cierre al mantener la Q."""

    def __init__(self):
        """Inicia en modo letras, esperando centrar la mano para armar el gesto."""
        self.modo = "letras"
        self.flechas_armada = False
        self.inicio_q = None
        self.cerrar = False
        self.estado = "Centra la mano para activar el cambio"

    def sin_mano(self):
        """Desarma el cambio y reinicia el temporizador si no se ve la mano."""
        self.flechas_armada = False
        self.inicio_q = None
        self.estado = "No se detecta la mano derecha"

    def actualizar(self, prediccion, posicion_x, instante):
        """Actualiza modo o temporizador a partir de la seña y zona de la mano.

        Args:
            prediccion: Etiqueta predicha para la mano derecha.
            posicion_x: Posición horizontal normalizada del centro de la palma.
            instante: Tiempo monotónico actual, en segundos.

        Returns:
            ``True`` si la seña Q se mantuvo durante el tiempo de cierre.
        """
        if prediccion.strip().lower() == "q":
            if self.inicio_q is None:
                self.inicio_q = instante
            self.flechas_armada = False
            tiempo_restante = max(
                0.0,
                DURACION_CIERRE_Q - (instante - self.inicio_q),
            )
            if tiempo_restante == 0:
                self.cerrar = True
                self.estado = "Cerrando..."
            else:
                self.estado = f"Mantén Q para cerrar: {tiempo_restante:.1f}s"
            return self.cerrar

        self.inicio_q = None
        if posicion_x < 1 / 3:
            zona = "izquierda"
        elif posicion_x > 2 / 3:
            zona = "derecha"
        else:
            zona = "centro"

        if zona == "centro":
            self.flechas_armada = True
            self.estado = "Gesto armado; desplaza la mano a un lado"
        elif self.flechas_armada:
            self.modo = "numeros" if zona == "izquierda" else "letras"
            self.flechas_armada = False
            self.estado = (
                "Modo números seleccionado"
                if self.modo == "numeros"
                else "Modo letras seleccionado"
            )
        else:
            self.estado = "Vuelve al centro para armar el cambio"
        return self.cerrar

    def progreso_q(self, instante):
        """Devuelve cuánto tiempo lleva manteniéndose la seña Q.

        Args:
            instante: Tiempo monotónico actual, en segundos.
        """
        if self.inicio_q is None:
            return 0.0
        return min(DURACION_CIERRE_Q, instante - self.inicio_q)


def iniciar():
    """Carga el modelo LSC y arranca una cámara compartida por ambos modos.

    Centrar la mano derecha y desplazarla a la izquierda selecciona números;
    desplazarla a la derecha selecciona letras. Mantener la seña Q por
    ``DURACION_CIERRE_Q`` segundos cierra el programa. Las teclas ``L`` y ``N``
    también permiten cambiar de modo manualmente.
    """
    clasificador = ClasificadorLSC.cargar()
    procesar_letras = crear_procesador(clasificador)
    control = ControlGestosManoDerecha()

    def procesar_imagen(imagen, resultados):
        """Controla gestos y envía el frame al procesador del modo activo.

        Args:
            imagen: Frame BGR espejado que se dibuja en pantalla.
            resultados: Landmarks y lateralidad de MediaPipe para ese frame.
        """
        alto, ancho = imagen.shape[:2]
        mano_derecha = None
        for puntos_mano, categorias in zip(
            resultados.hand_landmarks,
            resultados.handedness,
        ):
            if obtener_lado_espejado(categorias[0].category_name) == "Der":
                mano_derecha = puntos_mano
                break

        if mano_derecha is None:
            control.sin_mano()
        else:
            caracteristicas = normalizar_landmarks(
                mano_derecha,
                ancho,
                alto,
            )
            prediccion = clasificador.predecir(caracteristicas)
            centro_palma_x = sum(
                mano_derecha[indice].x
                for indice in (0, 5, 9, 13, 17)
            ) / 5
            control.actualizar(
                prediccion,
                centro_palma_x,
                time.monotonic(),
            )

        if control.modo == "letras":
            procesar_letras(imagen, resultados)
        else:
            procesar_numeros(imagen, resultados)

        cv2.line(
            imagen,
            (ancho // 3, 0),
            (ancho // 3, alto),
            (100, 100, 100),
            1,
        )
        cv2.line(
            imagen,
            (2 * ancho // 3, 0),
            (2 * ancho // 3, alto),
            (100, 100, 100),
            1,
        )
        instante = time.monotonic()
        if control.inicio_q is not None:
            cv2.putText(
                imagen,
                f"Q: {control.progreso_q(instante):.1f}/{DURACION_CIERRE_Q:.1f}s",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            imagen,
            f"Modo: {control.modo.upper()} | centro > izq: N | centro > der: L",
            (20, alto - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            imagen,
            control.estado,
            (20, alto - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    def procesar_tecla(tecla):
        """Cambia el modo si se pulsa ``L`` o ``N``.

        Args:
            tecla: Código de tecla entregado por OpenCV.

        Returns:
            ``True`` si la tecla fue atendida y ``False`` en caso contrario.
        """
        if tecla in (ord("l"), ord("L")):
            control.modo = "letras"
            control.flechas_armada = False
            return True
        if tecla in (ord("n"), ord("N")):
            control.modo = "numeros"
            control.flechas_armada = False
            return True
        return False

    ejecutar_detector(
        procesar_imagen,
        "LSC + numeros - L: letras | N: numeros | Q: salir",
        num_manos=2,
        procesar_tecla=procesar_tecla,
        debe_cerrar=lambda: control.cerrar,
    )


if __name__ == "__main__":
    iniciar()
