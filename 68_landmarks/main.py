# ================================
# Detector de Somnolencia con OpenCV, Dlib y Python
# ================================
# Autor: Alejandro Seminario Medina
# Optimizado para evitar variables globales
# Compatible con simpleaudio (requiere archivo .wav)
# ================================

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import simpleaudio as sa
import argparse
import imutils
import time
import dlib
import cv2

# --------------------------------
# Clase Detector de Somnolencia
# --------------------------------
class DrowsinessDetector:
    def __init__(self, predictor_path, alarm_path, webcam_index=0,
                 eye_ar_thresh=0.18, eye_ar_consec_frames=10,
                 text_alert="ALERTA: SOMNOLENCIA",
                 color_face=(0, 255, 0),
                 alarm_duration=5):
        """Inicializa parámetros, modelos y cámara."""
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.text_alert = text_alert
        self.color_face = color_face
        self.alarm_path = alarm_path
        self.alarm_duration = alarm_duration  # Duración en segundos

        self.counter = 0
        self.alarm_on = False

        # Inicializar detector y predictor
        print("[INFO] Cargando predictor de landmarks...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Índices de ojos
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # Cámara
        print("[INFO] Iniciando cámara...")
        self.vs = VideoStream(src=webcam_index).start()
        time.sleep(0.25)

    def sound_alarm(self, path, duracion=None):
        """Reproduce la alarma en un hilo separado usando simpleaudio."""
        try:
            wave_obj = sa.WaveObject.from_wave_file(path)
            play_obj = wave_obj.play()
            if duracion is None:
                duracion = self.alarm_duration
            time.sleep(duracion)
            play_obj.stop()
        except Exception as e:
            print(f"[ERROR] No se pudo reproducir el audio: {e}")
        finally:
            # Liberar el flag solo cuando la alarma terminó
            self.alarm_on = False

    @staticmethod
    def eye_aspect_ratio(eye):
        """Calcula el EAR de un ojo."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame, show_face=False):
        """Procesa un frame, detecta somnolencia y retorna frame anotado."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Dibujar contornos de ojos
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            if show_face:
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.color_face, 2)

            # Lógica de somnolencia
            if ear < self.eye_ar_thresh:
                self.counter += 1
                if self.counter >= self.eye_ar_consec_frames:
                    if not self.alarm_on:
                        self.alarm_on = True
                        if self.alarm_path:
                            t = Thread(target=self.sound_alarm, args=(self.alarm_path, self.alarm_duration))
                            t.daemon = True
                            t.start()
                    cv2.putText(frame, self.text_alert, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.counter = 0
                # ⚠️ Importante: NO tocar self.alarm_on aquí

            # Mostrar EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run(self, show_face=False):
        """Bucle principal."""
        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            frame = self.process_frame(frame, show_face)
            cv2.imshow("Detector de Somnolencia", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.stop()

# --------------------------------
# Script principal
# --------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat")
    ap.add_argument("-a", "--alarm", type=str, default="alarma.wav")  # Formato WAV obligatorio
    ap.add_argument("-w", "--webcam", type=int, default=0)
    ap.add_argument("--show-face", action="store_true")
    ap.add_argument("-d", "--duration", type=int, default=5, help="Duración de la alarma en segundos")
    args = vars(ap.parse_args())

    detector = DrowsinessDetector(
        predictor_path=args["shape_predictor"],
        alarm_path=args["alarm"],
        webcam_index=args["webcam"],
        alarm_duration=args["duration"]
    )
    detector.run(show_face=args["show_face"])