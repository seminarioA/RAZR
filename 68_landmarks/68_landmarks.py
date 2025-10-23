# ================================
# Visualización de Landmarks Faciales (68 puntos)
# ================================
# Autor: Alejandro Seminario Medina
# Adaptado y documentado en español
#
# Este script carga el predictor de landmarks faciales de dlib
# y muestra en pantalla los 68 puntos del rostro en tiempo real.

# --------------------------------
# Importación de librerías
# --------------------------------
from imutils import face_utils
from imutils.video import VideoStream
import argparse
import imutils
import time
import dlib
import cv2

# --------------------------------
# Manejo de argumentos de ejecución
# --------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
    help="Ruta al predictor de landmarks faciales (shape_predictor_68_face_landmarks.dat)")
ap.add_argument("-w", "--webcam", type=int, default=0,
    help="Índice de la webcam del sistema")
args = vars(ap.parse_args())

# --------------------------------
# Inicialización de detector y predictor facial
# --------------------------------
print("[INFO] Cargando predictor de landmarks faciales...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# --------------------------------
# Inicialización de la cámara
# --------------------------------
print("[INFO] Iniciando cámara...")
vs = VideoStream(src=args["webcam"]).start()
#time.sleep(0.5)

# --------------------------------
# Bucle principal de captura
# --------------------------------
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    rects = detector(gray, 0)

    for rect in rects:
        # Obtener landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Dibujar cada punto (68 en total)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Dibujar rectángulo alrededor del rostro
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar frame
    cv2.imshow("Landmarks Faciales - 68 puntos", frame)
    key = cv2.waitKey(1) & 0xFF

    # Salir con la tecla 'q'
    if key == ord("q"):
        break

# --------------------------------
# Liberar recursos
# --------------------------------
cv2.destroyAllWindows()
vs.stop()