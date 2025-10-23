import cv2
import mediapipe as mp

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
# Dibujar landmarks
# =========================
if results.multi_face_landmarks:
    h, w = image.shape[:2]
    for face_landmarks in results.multi_face_landmarks:
        # Ojo izquierdo en rojo
        for idx in LEFT_EYE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        # Ojo derecho en verde
        for idx in RIGHT_EYE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

# =========================
# Mostrar y guardar resultado
# =========================
cv2.imshow("Eye Landmarks", image)
cv2.imwrite("output_both_eyes.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()