from pathlib import Path

import numpy as np

INDICES_PALMA = (5, 17)
NUM_CARACTERISTICAS = 21 * 3
RUTA_MODELO = Path(__file__).parent / "models" / "lsc_knn.npz"


def normalizar_landmarks(puntos_mano, ancho, alto):
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
    def __init__(self, caracteristicas, etiquetas):
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
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontro el modelo LSC: {ruta}\n"
                "Ejecuta entrenar_lsc.py para generarlo."
            )
        with np.load(ruta, allow_pickle=False) as archivo:
            return cls(archivo["caracteristicas"], archivo["etiquetas"])

    def predecir(self, caracteristicas):
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
