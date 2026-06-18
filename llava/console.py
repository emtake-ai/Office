import cv2
import requests
import base64

RTSP_URL = "rtsp://admin:emtake145!@192.168.1.7:554/Streaming/Channels/101"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava:latest"


def frame_to_base64(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def ask_llava(question, frame):
    image_b64 = frame_to_base64(frame)

    payload = {
        "model": MODEL,
        "prompt": question,
        "images": [image_b64],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["response"]


def main():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        return

    print("RTSP + LLaVA console started.")
    print("Type your question and press ENTER.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame from camera.")
            continue

        try:
            answer = ask_llava(question, frame)
            print("\nLLaVA:", answer)

        except Exception as e:
            print("LLaVA error:", e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
