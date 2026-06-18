from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os

KNOWN_DIR = "known_faces"
THRESHOLD = 0.45

# Load known faces
known_faces = {}

for file in os.listdir(KNOWN_DIR):
    if file.endswith(".npy"):
        name = file[:-4]
        emb = np.load(os.path.join(KNOWN_DIR, file))
        emb = emb / np.linalg.norm(emb)
        known_faces[name] = emb

print("Loaded faces:")
for name in known_faces:
    print(" -", name)

# Face model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)  # CPU

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open /dev/video0")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        current = face.embedding
        current = current / np.linalg.norm(current)

        best_name = "Unknown"
        best_score = -1

        for name, known_emb in known_faces.items():

            score = np.dot(current, known_emb)

            if score > best_score:
                best_score = score
                best_name = name

        if best_score < THRESHOLD:
            best_name = "Unknown"

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

        label = f"{best_name} {best_score:.2f}"

        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow("Face Identification", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
