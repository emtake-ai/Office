from retinaface import RetinaFace
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = RetinaFace.detect_faces(frame)

    if isinstance(faces, dict):
        for key, face in faces.items():
            x1, y1, x2, y2 = face["facial_area"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for name, point in face["landmarks"].items():
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow("RetinaFace", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
