import cv2
import mediapipe as mp

# Inicializar Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Cargar imagen
image = cv2.imread("image.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Procesar imagen
results = face_mesh.process(rgb_image)

# Dibujar landmarks
if results.multi_face_landmarks:
    h, w = image.shape[:2]
    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# Mostrar y guardar resultado
cv2.imshow("Face Mesh", image)
cv2.imwrite("output_landmarks.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()