"""Entrena y evalúa el clasificador LSC a partir del CSV de landmarks."""

import csv
from collections import Counter

import numpy as np

from capturar_lsc import COLUMNAS, RUTA_DATOS
from lsc_clasificador import NUM_CARACTERISTICAS, RUTA_MODELO, ClasificadorLSC

RNG_SEMILLA = 42
FRACCION_PRUEBA = 0.2


def cargar_datos():
    """Lee el CSV y valida etiquetas, dimensiones y valores numéricos.

    Returns:
        Tupla ``(caracteristicas, etiquetas)`` en arreglos NumPy.

    Raises:
        FileNotFoundError: Si no existe ``datos_lsc.csv``.
        ValueError: Si el encabezado o alguna fila contiene datos inválidos.
    """
    if not RUTA_DATOS.is_file():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {RUTA_DATOS}")

    caracteristicas = []
    etiquetas = []
    with RUTA_DATOS.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames != COLUMNAS:
            raise ValueError("El encabezado de datos_lsc.csv no es valido")

        for numero_fila, fila in enumerate(lector, start=2):
            etiqueta = fila["etiqueta"].strip().lower()
            if not etiqueta:
                raise ValueError(f"Etiqueta vacia en la fila {numero_fila}")
            try:
                valores = [
                    float(fila[f"{eje}{indice}"])
                    for indice in range(21)
                    for eje in ("x", "y", "z")
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Landmarks invalidos en la fila {numero_fila}"
                ) from error
            if not np.isfinite(valores).all():
                raise ValueError(f"Valor no finito en la fila {numero_fila}")
            caracteristicas.append(valores)
            etiquetas.append(etiqueta)

    if len(caracteristicas) == 0:
        raise ValueError("No hay ejemplos en datos_lsc.csv")
    return (
        np.asarray(caracteristicas, dtype=np.float32),
        np.asarray(etiquetas, dtype=str),
    )


def evaluar(caracteristicas, etiquetas):
    """Mide exactitud con una división aleatoria estratificada por etiqueta.

    Args:
        caracteristicas: Matriz de ejemplos, con 63 valores por fila.
        etiquetas: Etiquetas alineadas fila por fila con la matriz.

    Returns:
        Tupla con la proporción de aciertos y la cantidad de ejemplos de prueba.
    """
    rng = np.random.default_rng(RNG_SEMILLA)
    indices_entrenamiento = []
    indices_prueba = []
    for etiqueta in sorted(set(etiquetas)):
        indices = np.flatnonzero(etiquetas == etiqueta)
        rng.shuffle(indices)
        if len(indices) < 2:
            indices_entrenamiento.extend(indices)
            continue
        cantidad_prueba = max(1, round(len(indices) * FRACCION_PRUEBA))
        indices_prueba.extend(indices[:cantidad_prueba])
        indices_entrenamiento.extend(indices[cantidad_prueba:])

    if not indices_prueba:
        raise ValueError("Se necesitan al menos dos ejemplos para evaluar")
    if not indices_entrenamiento:
        raise ValueError("No hay ejemplos disponibles para entrenamiento")

    clasificador = ClasificadorLSC(
        caracteristicas[indices_entrenamiento],
        etiquetas[indices_entrenamiento],
    )
    aciertos = sum(
        clasificador.predecir(caracteristicas[indice]) == etiquetas[indice]
        for indice in indices_prueba
    )
    return aciertos / len(indices_prueba), len(indices_prueba)


def entrenar():
    """Evalúa todos los datos y guarda el modelo final en formato NPZ."""
    caracteristicas, etiquetas = cargar_datos()
    if caracteristicas.shape[1] != NUM_CARACTERISTICAS:
        raise ValueError("Cada ejemplo debe contener 63 landmarks normalizados")

    precision, cantidad_prueba = evaluar(caracteristicas, etiquetas)
    conteos = Counter(etiquetas)
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RUTA_MODELO,
        caracteristicas=caracteristicas,
        etiquetas=etiquetas,
    )

    print(f"Modelo guardado en: {RUTA_MODELO}")
    print(f"Ejemplos para el modelo: {len(etiquetas)}")
    print(f"Etiquetas: {len(conteos)}")
    print(f"Ejemplos de prueba: {cantidad_prueba}")
    print(f"Precision de prueba aleatoria: {precision:.1%}")
    print("Ejemplos por etiqueta:")
    for etiqueta, cantidad in sorted(conteos.items()):
        print(f"  {etiqueta}: {cantidad}")
    print(
        "Nota: la precision puede ser optimista si las muestras de prueba "
        "se parecen a ejemplos capturados consecutivamente."
    )


if __name__ == "__main__":
    entrenar()
