# -*- coding: utf-8 -*-
"""
RAZR - Detector de Somnolencia (versión IEEE Q1 mejorada)
Autor: Alejandro Valentino
Modificaciones:
 - Umbral EAR = 0.16
 - Gráfico EAR con ejes IEEE Q1 (rango 0.0–0.4)
 - Botones:
      g → guardar histograma (Fig. 2, .jpg y .png)
      f → guardar rostro con landmarks oculares (Fig. 1)
 - Marcado ocular con puntos rojos
"""

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from collections import deque
import simpleaudio as sa
import mediapipe as mp
import matplotlib.pyplot as plt
import imutils
import argparse
import cv2
import time
import numpy as np
import csv
import os
from typing import Optional, List, Tuple, Dict

try:
    from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ======================================================
# Clase principal
# ======================================================
class DetectorDeSomnolencia:
    def __init__(self,
                 ruta_alarma: str,
                 indice_camara: int = 0,
                 umbral_ear: float = 0.16,
                 tiempo_umbral: float = 1.5,
                 texto_alerta: str = "ALERTA: SOMNOLENCIA",
                 duracion_alarma: int = 5,
                 ancho_frame: int = 450,
                 mostrar_grafico: bool = True,
                 mostrar_fps_origen: bool = False,
                 mostrar_fps_procesado: bool = False,
                 audio_activo: bool = True,
                 blink_umbral: float = 0.21,
                 max_duracion_parpadeo: float = 0.5,
                 salida_dir: str = "results"):

        self.umbral_ear = umbral_ear
        self.tiempo_umbral = tiempo_umbral
        self.texto_alerta = texto_alerta
        self.duracion_alarma = duracion_alarma
        self.ancho_frame = ancho_frame
        self.mostrar_grafico = mostrar_grafico
        self.mostrar_fps_origen = mostrar_fps_origen
        self.mostrar_fps_procesado = mostrar_fps_procesado
        self.audio_activo = audio_activo
        self.ruta_alarma = ruta_alarma
        self.blink_umbral = blink_umbral
        self.max_duracion_parpadeo = max_duracion_parpadeo

        # Estado
        self.alarma_activa = False
        self.tiempo_inicio = None
        self.obj_audio = None
        self.historial_ear = deque()

        self.indices_ojo_izq = [33, 160, 158, 133, 153, 144]
        self.indices_ojo_der = [362, 385, 387, 263, 373, 380]

        # Inicialización
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self.vs = VideoStream(src=indice_camara).start()
        time.sleep(0.25)
        self.fps_origen = self._obtener_fps(indice_camara)

        if mostrar_grafico:
            self._inicializar_grafico()

        os.makedirs(salida_dir, exist_ok=True)
        self.salida_dir = salida_dir
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(salida_dir, f"metrics_run_{ts}.csv")

        self.registros: List[Dict] = []
        self.frame_ultimo = None  # para guardar foto

    # -------------------------
    def _obtener_fps(self, indice_camara: int) -> int:
        cap = cv2.VideoCapture(indice_camara)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 0
        cap.release()
        return fps

    # -------------------------
    def _inicializar_grafico(self):
        plt.ion()
        self.figura, self.ax = plt.subplots(figsize=(6.4, 3.2))
        self.linea_ear, = self.ax.plot([], [], '-', label='EAR', linewidth=1.2, color='blue')
        self.ax.axhline(y=self.umbral_ear, color='red', linestyle='--',
                        linewidth=1.0, label=f'Umbral adaptativo θ ≈ {self.umbral_ear:.2f}')
        self.ax.set(
            xlim=(0, 60), ylim=(0.0, 0.4),
            xlabel="Tiempo transcurrido (s, últimos 60 s)",
            ylabel="Eye Aspect Ratio (EAR)",
            title="Evolución temporal del Eye Aspect Ratio (EAR) en tiempo real"
        )
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        self.ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show(block=False)

    # -------------------------
    @staticmethod
    def _calcular_ear(ojo):
        A = dist.euclidean(ojo[1], ojo[5])
        B = dist.euclidean(ojo[2], ojo[4])
        C = dist.euclidean(ojo[0], ojo[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    # -------------------------
    def _actualizar_grafico(self, ear: float):
        t_actual = time.time()
        self.historial_ear.append((t_actual, ear))
        while self.historial_ear and (t_actual - self.historial_ear[0][0]) > 60:
            self.historial_ear.popleft()

        if not self.mostrar_grafico:
            return

        tiempos_rel = [t - self.historial_ear[0][0] for t, _ in self.historial_ear]
        valores = [v for _, v in self.historial_ear]
        self.linea_ear.set_data(tiempos_rel, valores)
        self.ax.set_xlim(0, 60)
        self.ax.set_ylim(0.0, 0.4)
        self.figura.canvas.draw()
        self.figura.canvas.flush_events()

    # -------------------------
    def _activar_alarma(self):
        if not self.audio_activo or self.alarma_activa:
            return
        try:
            self.obj_audio = sa.WaveObject.from_wave_file(self.ruta_alarma).play()
            self.alarma_activa = True
        except Exception:
            self.alarma_activa = False

    # -------------------------
    def _gestionar_audio(self):
        if self.obj_audio and not self.obj_audio.is_playing():
            self.alarma_activa = False
            self.obj_audio = None

    # -------------------------
    def _procesar_frame(self, frame, frame_idx, tiempo_previo_proc):
        t_inicio_proc = time.time()
        resultados = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        ear_prom = None

        if resultados.multi_face_landmarks:
            for landmarks in resultados.multi_face_landmarks:
                ojo_izq = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
                           for i in self.indices_ojo_izq]
                ojo_der = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
                           for i in self.indices_ojo_der]

                ear_izq = self._calcular_ear(ojo_izq)
                ear_der = self._calcular_ear(ojo_der)
                ear_prom = (ear_izq + ear_der) / 2.0

                # Dibujar puntos rojos (IEEE Fig. 1)
                for (x, y) in ojo_izq + ojo_der:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                if ear_prom < self.umbral_ear:
                    if self.tiempo_inicio is None:
                        self.tiempo_inicio = time.time()
                    elif (time.time() - self.tiempo_inicio) >= self.tiempo_umbral:
                        self._activar_alarma()
                        cv2.putText(frame, self.texto_alerta, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.tiempo_inicio = None

                cv2.putText(frame, f"EAR: {ear_prom:.3f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.frame_ultimo = frame.copy()

        if ear_prom is not None:
            self._actualizar_grafico(ear_prom)

        tiempo_fin_proc = time.time()
        latency_ms = (tiempo_fin_proc - t_inicio_proc) * 1000.0
        registro = {
            "frame_index": frame_idx,
            "timestamp": tiempo_fin_proc,
            "ear": ear_prom,
            "alarm": int(self.alarma_activa),
            "latency_ms": latency_ms
        }
        self.registros.append(registro)
        return frame, tiempo_fin_proc

    # -------------------------
    def _guardar_csv_registros(self):
        keys = ["frame_index", "timestamp", "ear", "alarm", "latency_ms"]
        with open(self.csv_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.registros:
                writer.writerow({k: r.get(k, None) for k in keys})

    # -------------------------
    def ejecutar(self):
        print("[INFO] Presiona 'f' para guardar rostro (Fig.1), 'g' para guardar histograma (Fig.2), 'q' para salir.")
        tiempo_previo = None
        frame_idx = 0
        while True:
            frame_raw = self.vs.read()
            if frame_raw is None:
                break
            frame = imutils.resize(frame_raw, width=self.ancho_frame)
            frame, tiempo_previo = self._procesar_frame(frame, frame_idx, tiempo_previo)

            cv2.imshow("RAZR - Detector de Somnolencia (IEEE Q1)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("f"):
                if self.frame_ultimo is not None:
                    path_fig1 = os.path.join(self.salida_dir, "fig1_rostro_EAR.png")
                    cv2.imwrite(path_fig1, self.frame_ultimo)
                    print(f"[FIG.1] Rostro guardado en {path_fig1}")

            elif key == ord("g"):
                path_fig2_png = os.path.join(self.salida_dir, "fig2_histograma_EAR.png")
                path_fig2_jpg = os.path.join(self.salida_dir, "fig2_histograma_EAR.jpg")
                self.figura.savefig(path_fig2_png, dpi=600, bbox_inches='tight', transparent=False)
                self.figura.savefig(path_fig2_jpg, dpi=600, bbox_inches='tight', transparent=False)
                print(f"[FIG.2] Histograma guardado en {path_fig2_png} y {path_fig2_jpg}")

            elif key == ord("q"):
                break

            frame_idx += 1

        self._guardar_csv_registros()
        self._liberar_recursos()
        print("[INFO] Ejecución finalizada. CSV guardado.")

    # -------------------------
    def _liberar_recursos(self):
        cv2.destroyAllWindows()
        try:
            plt.close('all')
        except Exception:
            pass
        try:
            self.vs.stop()
        except Exception:
            try:
                self.vs.release()
            except Exception:
                pass
        try:
            self.face_mesh.close()
        except Exception:
            pass


# ======================================================
# Ejecución principal
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAZR - Detector de Somnolencia (IEEE Q1)")
    parser.add_argument("--ruta-alarma", type=str, default="alarma.wav")
    parser.add_argument("--indice-camara", type=int, default=0)
    parser.add_argument("--umbral-ear", type=float, default=0.16)  # ← Umbral modificado
    parser.add_argument("--tiempo-umbral", type=float, default=1.0)
    parser.add_argument("--duracion-alarma", type=int, default=5)
    parser.add_argument("--texto-alerta", type=str, default="ALERTA: SOMNOLENCIA")
    parser.add_argument("--mostrar-grafico", action="store_true")
    parser.add_argument("--audio-activo", action="store_true")
    parser.add_argument("--salida-dir", type=str, default="results")
    args = parser.parse_args()

    detector = DetectorDeSomnolencia(
        ruta_alarma=args.ruta_alarma,
        indice_camara=args.indice_camara,
        umbral_ear=args.umbral_ear,
        tiempo_umbral=args.tiempo_umbral,
        duracion_alarma=args.duracion_alarma,
        texto_alerta=args.texto_alerta,
        mostrar_grafico=args.mostrar_grafico,
        audio_activo=args.audio_activo,
        salida_dir=args.salida_dir
    )
    detector.ejecutar()
