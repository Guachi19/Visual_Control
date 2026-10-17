"""Normalización de landmarks y clasificador de vecinos cercanos para LSC."""

from pathlib import Path

import numpy as np

INDICES_PALMA = (5, 17)
NUM_CARACTERISTICAS = 21 * 3
RUTA_MODELO = Path(__file__).parent / "models" / "lsc_knn.npz"


def normalizar_landmarks(puntos_mano, ancho, alto):
    """Normaliza los 21 landmarks respecto a muñeca y ancho de palma.

    Args:
        puntos_mano: Secuencia de 21 puntos normalizados de MediaPipe, cada uno
            con coordenadas ``x``, ``y`` y ``z``.
        ancho: Ancho del frame en píxeles.
        alto: Alto del frame en píxeles.

    Returns:
        Vector NumPy de 63 características normalizadas.

    Raises:
        ValueError: Si no se puede calcular una escala positiva para la palma.
    """
    muneca = puntos_mano[0]
    base_indice = puntos_mano[INDICES_PALMA[0]]
    base_menique = puntos_mano[INDICES_PALMA[1]]
    escala = (
        ((base_indice.x - base_menique.x) * ancho) ** 2
        + ((base_indice.y - base_menique.y) * alto) ** 2
    ) ** 0.5
    if escala <= 0:
        raise ValueError("No se pudo calcular el tamano de la palma")

    caracteristicas = []
    for punto in puntos_mano:
        caracteristicas.extend(
            (
                (punto.x - muneca.x) * ancho / escala,
                (punto.y - muneca.y) * alto / escala,
                (punto.z - muneca.z) * ancho / escala,
            )
        )
    return np.asarray(caracteristicas, dtype=np.float32)


class ClasificadorLSC:
    """Clasificador 1-NN que compara landmarks con ejemplos etiquetados."""

    def __init__(self, caracteristicas, etiquetas):
        """Inicializa el clasificador con los ejemplos de entrenamiento.

        Args:
            caracteristicas: Matriz NumPy de forma ``(ejemplos, 63)``.
            etiquetas: Etiqueta asociada a cada fila de características.

        Raises:
            ValueError: Si la matriz no tiene 63 columnas o no hay etiquetas.
        """
        if caracteristicas.ndim != 2 or caracteristicas.shape[1] != NUM_CARACTERISTICAS:
            raise ValueError("El modelo no tiene 63 caracteristicas por ejemplo")
        if len(caracteristicas) != len(etiquetas) or len(etiquetas) == 0:
            raise ValueError("El modelo no contiene ejemplos etiquetados")
        self.caracteristicas = caracteristicas.astype(np.float32, copy=False)
        self.etiquetas = etiquetas.astype(str, copy=False)
        self.normas = np.einsum(
            "ij,ij->i",
            self.caracteristicas,
            self.caracteristicas,
        )

    @classmethod
    def cargar(cls, ruta=RUTA_MODELO):
        """Carga la matriz de ejemplos y etiquetas desde un archivo NPZ.

        Args:
            ruta: Ruta del modelo comprimido; por defecto usa ``models``.

        Raises:
            FileNotFoundError: Si no existe el modelo indicado.
        """
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontro el modelo LSC: {ruta}\n"
                "Ejecuta entrenar_lsc.py para generarlo."
            )
        with np.load(ruta, allow_pickle=False) as archivo:
            return cls(archivo["caracteristicas"], archivo["etiquetas"])

    def predecir(self, caracteristicas):
        """Devuelve la etiqueta del ejemplo de entrenamiento más cercano.

        Args:
            caracteristicas: Vector de 63 valores normalizados.

        Returns:
            La etiqueta del vecino más cercano.

        Raises:
            ValueError: Si el vector no tiene exactamente 63 valores.
        """
        vector = np.asarray(caracteristicas, dtype=np.float32)
        if vector.shape != (NUM_CARACTERISTICAS,):
            raise ValueError("Se esperaban 63 caracteristicas para predecir")
        distancias = (
            self.normas
            + np.dot(vector, vector)
            - 2 * np.dot(self.caracteristicas, vector)
        )
        indice = int(np.argmin(distancias))
        return self.etiquetas[indice]
