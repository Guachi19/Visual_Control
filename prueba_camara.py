"""Prueba mínima para confirmar que la cámara puede abrirse."""

from detector_comun import ejecutar_camara


if __name__ == "__main__":
    ejecutar_camara(lambda imagen: None, "Prueba de camara")
