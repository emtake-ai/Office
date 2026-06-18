import cv2

RTSP_URL = "rtsp://admin:emtake145!@192.168.1.7:554/Streaming/Channels/101"

def main():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        return

    print("RTSP live display started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        if not ret:
            print("Failed to read frame.")
            break

        cv2.imshow("RTSP Live Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
