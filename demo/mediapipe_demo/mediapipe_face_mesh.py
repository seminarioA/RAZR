# ================================
# Visualización completa de MediaPipe Face Mesh
# ================================
# Autor: Alejandro Seminario Medina
# Muestra todos los puntos faciales (468) y un bounding box
# ================================

from imutils.video import VideoStream
import mediapipe as mp
import argparse
import imutils
import time
import cv2
import numpy as np

class FaceMeshVisualizer:
    def __init__(self, webcam_index=0):
        # Inicializar MediaPipe FaceMesh
        print("[INFO] Cargando MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Cámara
        print("[INFO] Iniciando cámara...")
        self.vs = VideoStream(src=webcam_index).start()
        time.sleep(0.25)

    def process_frame(self, frame):
        """Procesa un frame, dibuja landmarks y bounding box."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]

            for face_landmarks in results.multi_face_landmarks:
                puntos = []

                # Dibujar los 468 puntos faciales
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    puntos.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Calcular bounding box
                puntos_np = np.array(puntos)
                x_min, y_min = np.min(puntos_np, axis=0)
                x_max, y_max = np.max(puntos_np, axis=0)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 1)

        return frame

    def run(self):
        """Bucle principal."""
        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=600)
            frame = self.process_frame(frame)
            cv2.imshow("Face Mesh: 468 puntos + Bounding Box", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.stop()

# --------------------------------
# Script principal
# --------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0)
    args = vars(ap.parse_args())

    visualizer = FaceMeshVisualizer(webcam_index=args["webcam"])
    visualizer.run()