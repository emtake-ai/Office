import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import json
import time
import base64
import threading
import requests
import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


RTSP_URL = "rtsp://admin:emtake145!@192.168.1.7:554/Streaming/Channels/101"

YOLO_MODEL_PATH = "yolo11n.pt"
VECTOR_DB_PATH = "./embedding/office_embeddings.json"
KNOWN_DIR = "./known_faces"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
LLAVA_MODEL = "llava:latest"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MONITOR_INTERVAL = 1.0
TOP_K = 3
FACE_THRESHOLD = 0.45


latest_object_counts = {}
latest_face_names = []
latest_face_images = {}
latest_frame = None
latest_annotated_frame = None

state_lock = threading.Lock()
stop_event = threading.Event()


FURNITURE_OBJECTS = {"chair", "bench", "couch", "table", "desk", "dining table"}
UTILITY_OBJECTS = {
    "laptop", "keyboard", "mouse", "monitor", "tv", "cell phone",
    "remote", "book", "clock", "mic", "microphone", "speaker",
    "camera", "printer"
}
WORK_OBJECTS = {"laptop", "keyboard", "mouse", "monitor", "tv", "desk", "chair"}


def load_known_faces():
    known_faces = {}

    if not os.path.exists(KNOWN_DIR):
        return known_faces

    for file in os.listdir(KNOWN_DIR):
        if file.endswith(".npy"):
            name = file[:-4]
            emb = np.load(os.path.join(KNOWN_DIR, file))
            emb = emb / np.linalg.norm(emb)
            known_faces[name] = emb

    return known_faces


def recognize_faces(face_app, known_faces, frame):
    faces = face_app.get(frame)
    names = []
    face_images = {}

    h, w = frame.shape[:2]

    for idx, face in enumerate(faces, start=1):
        current = face.embedding
        current = current / np.linalg.norm(current)

        best_name = "Unknown"
        best_score = -1.0

        for name, known_emb in known_faces.items():
            score = float(np.dot(current, known_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < FACE_THRESHOLD:
            best_name = "Unknown"

        names.append(best_name)

        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2].copy()

            key = best_name
            if key in face_images:
                key = f"{best_name}_{idx}"

            face_images[key] = crop

    return names, face_images


def is_llava_request(question):
    return question.lower().startswith("llava,")


def extract_llava_question(question):
    parts = question.split(",", 1)
    if len(parts) < 2:
        return "Describe the current image."
    q = parts[1].strip()
    return q if q else "Describe the current image."


def frame_to_base64(frame):
    small = cv2.resize(frame, (480, 360))
    ok, buffer = cv2.imencode(
        ".jpg",
        small,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )

    if not ok:
        return None

    return base64.b64encode(buffer).decode("utf-8")


def ask_llava(question, frame):
    image_b64 = frame_to_base64(frame)

    if image_b64 is None:
        return "Failed to encode image."

    payload = {
        "model": LLAVA_MODEL,
        "prompt": question,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 180
        }
    }

    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=300
    )
    res.raise_for_status()

    return res.json().get("response", "").strip()


def is_image_request(question):
    q = question.lower()
    return any(k in q for k in [
        "can i get image", "show image", "display image",
        "get image", "show camera", "show frame",
        "show me image", "camera image", "image please",
        "사진", "이미지", "카메라 보여", "화면 보여",
    ])


def is_who_request(question):
    q = question.lower()
    return any(k in q for k in [
        "who is there", "who are there", "may i know who is there",
        "who do you see", "who is in office", "who is in the office",
        "who is here", "누가 있어", "누구 있어", "누구야", "누가 있니",
    ])


def get_embedding(text):
    payload = {"model": EMBED_MODEL, "prompt": text}
    res = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=120)
    res.raise_for_status()
    return res.json()["embedding"]


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def load_vector_db():
    if not os.path.exists(VECTOR_DB_PATH):
        return []

    with open(VECTOR_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_context(query, records, top_k=TOP_K):
    if not records:
        return ""

    try:
        query_embedding = get_embedding(query)
    except Exception:
        return ""

    scored = []

    for record in records:
        emb = record.get("embedding")
        if emb is None:
            continue

        score = cosine_similarity(query_embedding, emb)
        scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)

    blocks = []

    for i, (score, record) in enumerate(scored[:top_k], start=1):
        blocks.append(
            f"""
[RAG_CHUNK_{i}]
score: {score:.4f}
source: {record.get("source", "unknown")}
chunk_index: {record.get("chunk_index", "unknown")}
content:
{record.get("text", "")}
""".strip()
        )

    return "\n\n".join(blocks)


def detect_objects(yolo, frame):
    results = yolo.predict(frame, device="cpu", verbose=False)
    result = results[0]

    object_counts = {}

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = yolo.names.get(cls_id, str(cls_id))
            object_counts[cls_name] = object_counts.get(cls_name, 0) + 1

    annotated = result.plot()
    return object_counts, annotated


def capture_frame(cap):
    ret, frame = cap.read()

    if not ret:
        return None

    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))


def format_detected_objects(object_counts):
    if not object_counts:
        return "none"

    return "\n".join(
        f"{name}: {count}"
        for name, count in sorted(object_counts.items())
    )


def format_faces(face_names):
    if not face_names:
        return "none"

    counts = {}

    for name in face_names:
        counts[name] = counts.get(name, 0) + 1

    return "\n".join(f"{name}: {count}" for name, count in counts.items())


def build_ontology(object_counts, face_names):
    person_count = object_counts.get("person", 0)

    facts = []
    types = []
    relations = []
    contexts = []

    for obj, count in sorted(object_counts.items()):
        facts.append(f"{obj}({count})")

        if obj == "person":
            for i in range(1, count + 1):
                types.append(f"person_{i}: person")
        else:
            types.append(f"{obj}: object")

    if face_names:
        for i, name in enumerate(face_names, start=1):
            facts.append(f"face_name({name})")
            relations.append(f"person_{i} identity is {name}")

    furniture = [o for o in object_counts if o in FURNITURE_OBJECTS]
    utility = [o for o in object_counts if o in UTILITY_OBJECTS]
    work_objs = [o for o in object_counts if o in WORK_OBJECTS]

    if furniture:
        contexts.append("office_furniture_detected")
        for obj in furniture:
            relations.append(f"{obj} is office furniture")

    if utility:
        contexts.append("office_utility_detected")
        for obj in utility:
            relations.append(f"{obj} is office utility")

    if person_count > 0:
        contexts.append("office_occupied")
    else:
        contexts.append("office_empty")

    if person_count > 0 and work_objs:
        contexts.append("office_working_detected")
        for i in range(1, person_count + 1):
            name = face_names[i - 1] if i - 1 < len(face_names) else f"person_{i}"
            relations.append(f"{name} is working")

    if person_count > 0 or furniture or utility:
        contexts.append("office_environment_detected")

    return f"""
[FACTS]
{chr(10).join(facts) if facts else "none"}

[TYPES]
{chr(10).join(types) if types else "none"}

[RELATIONS]
{chr(10).join(relations) if relations else "none"}

[CONTEXT]
{chr(10).join(contexts) if contexts else "none"}
""".strip()


def object_list_text(objects, object_counts):
    return ", ".join(f"{object_counts[obj]} {obj}(s)" for obj in objects)


def direct_answer(question, object_counts, face_names, ontology_text):
    q = question.lower()

    person_count = object_counts.get("person", 0)

    furniture = [o for o in object_counts if o in FURNITURE_OBJECTS]
    utility = [o for o in object_counts if o in UTILITY_OBJECTS]
    non_person_objects = [o for o in object_counts if o != "person"]

    if is_who_request(question) or "name" in q:
        known = [n for n in face_names if n != "Unknown"]

        if known:
            return "Detected person name(s): " + ", ".join(known) + "."

        if face_names:
            return "Unknown person is there."

        return "No face is detected."

    if "environment" in q or "situation" in q or "current" in q:
        parts = []

        if person_count > 0:
            parts.append(f"{person_count} person(s) detected")

        known = [n for n in face_names if n != "Unknown"]
        if known:
            parts.append("names: " + ", ".join(known))

        if furniture:
            parts.append("furniture: " + object_list_text(furniture, object_counts))

        if utility:
            parts.append("utility: " + object_list_text(utility, object_counts))

        if "office_working_detected" in ontology_text:
            parts.append("office work detected")

        if parts:
            return "Office situation: " + "; ".join(parts) + "."

        return "No clear office situation is detected."

    if "furniture" in q:
        if furniture:
            return "Office furniture: " + object_list_text(furniture, object_counts) + "."
        return "No office furniture is detected."

    if "utility" in q or "device" in q or "equipment" in q:
        if utility:
            return "Office utility: " + object_list_text(utility, object_counts) + "."
        return "No office utility is detected."

    if "with person" in q or "nearby person" in q or "with them" in q:
        if non_person_objects:
            return "Objects with person: " + ", ".join(non_person_objects) + "."
        return "No object is detected with person."

    if "working" in q or "work" in q:
        if "office_working_detected" in ontology_text:
            known = [n for n in face_names if n != "Unknown"]

            if known:
                return ", ".join(known) + " is working."

            return f"Yes, {person_count} person(s) are working."

        return "Working is not detected."

    for obj, count in object_counts.items():
        obj_l = obj.lower()

        if obj_l in q:
            if "how many" in q or "count" in q:
                if obj_l == "person":
                    return f"{count} person(s) are detected."
                return f"{count} {obj_l}(s) are detected."

            if "is there" in q or "are there" in q:
                return f"Yes, {count} {obj_l}(s) are detected."

    return None


def ask_llama(question, detected_text, face_text, ontology_text, rag_context):
    prompt = f"""
You are an office sensor assistant.

Use only:
1. YOLO detected objects
2. Face recognition names
3. Ontology
4. RAG context

Rules:
- Do not invent objects.
- Do not invent people.
- Do not invent activity unless ontology gives it.
- If face name is known, use the name.
- Keep answer short and direct.
- Do not ask another question.
- If you do not know, say "I do not know."

[YOLO_OBJECTS]
{detected_text}

[FACE_RECOGNITION]
{face_text}

[ONTOLOGY]
{ontology_text}

[RAG_CONTEXT]
{rag_context}

[USER_QUESTION]
{question}

Answer only:
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "num_predict": 100
        }
    }

    res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    res.raise_for_status()

    return res.json().get("response", "").strip()


def show_image_window(frame, annotated_frame=None):
    if annotated_frame is not None:
        cv2.imshow("YOLO Detection Image", annotated_frame)
    else:
        cv2.imshow("Camera Image", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_face_windows(face_images):
    if not face_images:
        return

    for name, crop in face_images.items():
        if crop is not None and crop.size > 0:
            cv2.imshow(f"Detected Face - {name}", crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_alert(message):
    print(f"\nAssistant: {message}")
    print("You: ", end="", flush=True)


def monitor_loop(cap, yolo, face_app, known_faces):
    global latest_object_counts
    global latest_face_names
    global latest_face_images
    global latest_frame
    global latest_annotated_frame

    prev_person_count = 0
    prev_known_names = set()

    while not stop_event.is_set():
        frame = capture_frame(cap)

        if frame is None:
            time.sleep(MONITOR_INTERVAL)
            continue

        try:
            object_counts, annotated = detect_objects(yolo, frame)
            face_names, face_images = recognize_faces(face_app, known_faces, frame)

        except Exception:
            time.sleep(MONITOR_INTERVAL)
            continue

        with state_lock:
            latest_object_counts = object_counts
            latest_face_names = face_names
            latest_face_images = face_images
            latest_frame = frame
            latest_annotated_frame = annotated

        current_person_count = object_counts.get("person", 0)
        current_known_names = set([n for n in face_names if n != "Unknown"])

        if prev_person_count == 0 and current_person_count > 0:
            print_alert("new person is there")

        if prev_person_count > 0 and current_person_count == 0:
            print_alert("person disappeared")

        new_names = current_known_names - prev_known_names

        for name in new_names:
            print_alert(f"{name} is there")

        prev_person_count = current_person_count
        prev_known_names = current_known_names

        time.sleep(MONITOR_INTERVAL)


def main():
    print("Assistant: Loading YOLO on CPU...")
    yolo = YOLO(YOLO_MODEL_PATH)
    yolo.to("cpu")

    print("Assistant: Loading known faces...")
    known_faces = load_known_faces()

    print("Assistant: Loading face recognition model on CPU...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=-1, det_size=(128, 128))

    records = load_vector_db()

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Assistant: Failed to open RTSP stream.")
        return

    monitor_thread = threading.Thread(
        target=monitor_loop,
        args=(cap, yolo, face_app, known_faces),
        daemon=True
    )
    monitor_thread.start()

    print("Assistant: Practice system started. Type a question. Type 'exit' to stop.")
    print("Assistant: Use 'llava, your question' to ask LLaVA about the current image.")

    try:
        while True:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit"]:
                break

            with state_lock:
                object_counts = latest_object_counts.copy()
                face_names = list(latest_face_names)
                face_images = dict(latest_face_images)
                frame = latest_frame.copy() if latest_frame is not None else None
                annotated = (
                    latest_annotated_frame.copy()
                    if latest_annotated_frame is not None
                    else None
                )

            if frame is None:
                print("Assistant: No camera frame is ready yet.")
                continue

            if is_llava_request(question):
                llava_question = extract_llava_question(question)

                try:
                    answer = ask_llava(llava_question, frame)
                except Exception:
                    answer = "LLaVA is not available. Check: ollama pull llava:latest"

                print("Assistant:", answer)
                continue

            detected_text = format_detected_objects(object_counts)
            face_text = format_faces(face_names)
            ontology_text = build_ontology(object_counts, face_names)

            if is_image_request(question):
                show_image_window(frame, annotated)
                print("Assistant: Image displayed.")
                continue

            if is_who_request(question):
                show_face_windows(face_images)

                known = [n for n in face_names if n != "Unknown"]

                if known:
                    print("Assistant:", ", ".join(known), "is there.")
                elif face_names:
                    print("Assistant: Unknown person is there.")
                else:
                    print("Assistant: No face is detected.")

                continue

            direct = direct_answer(question, object_counts, face_names, ontology_text)

            if direct is not None:
                print("Assistant:", direct)
                continue

            rag_query = f"""
Question:
{question}

YOLO:
{detected_text}

Faces:
{face_text}

Ontology:
{ontology_text}
"""

            rag_context = retrieve_context(rag_query, records, top_k=TOP_K)

            try:
                answer = ask_llama(
                    question=question,
                    detected_text=detected_text,
                    face_text=face_text,
                    ontology_text=ontology_text,
                    rag_context=rag_context
                )

            except Exception as e:
                answer = f"LLaVA error: {e}"
            print("Assistant:", answer)

    except KeyboardInterrupt:
        print("\nAssistant: Stopped.")

    finally:
        stop_event.set()
        monitor_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
