import cv2

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

while True:
    ok, imagen = camara.read()
    imagen = cv2.flip(imagen, 1)
    if not ok:
        break
    cv2.imshow("Prueba de camara", imagen)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()