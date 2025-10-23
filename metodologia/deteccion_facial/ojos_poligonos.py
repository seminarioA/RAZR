import cv2
import mediapipe as mp
import numpy as np

# =========================
# Inicializar Face Mesh
# =========================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =========================
# Cargar imagen
# =========================
image = cv2.imread("image.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =========================
# Procesar imagen
# =========================
results = face_mesh.process(rgb_image)

# =========================
# Landmarks de ambos ojos
# =========================
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# =========================
# Dibujar polígonos de los ojos
# =========================
if results.multi_face_landmarks:
    h, w = image.shape[:2]
    for face_landmarks in results.multi_face_landmarks:
        # Ojo izquierdo
        left_eye_points = []
        for idx in LEFT_EYE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            left_eye_points.append((x, y))
        left_eye_array = np.array(left_eye_points, np.int32)
        cv2.polylines(image, [left_eye_array], isClosed=True, color=(0, 0, 255), thickness=2)

        # Ojo derecho
        right_eye_points = []
        for idx in RIGHT_EYE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            right_eye_points.append((x, y))
        right_eye_array = np.array(right_eye_points, np.int32)
        cv2.polylines(image, [right_eye_array], isClosed=True, color=(0, 0, 255), thickness=2)

# =========================
# Mostrar y guardar resultado
# =========================
cv2.imshow("Eye Polygons", image)
cv2.imwrite("output_eye_polygons.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()