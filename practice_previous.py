import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import json
import requests
import cv2
import numpy as np
from ultralytics import YOLO


RTSP_URL = "rtsp://admin:emtake145!@192.168.1.7:554/Streaming/Channels/101"

YOLO_MODEL_PATH = "yolo11n.pt"
VECTOR_DB_PATH = "./office_embeddings.json"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1B"

TOP_K = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def is_image_request(question):
    q = question.lower()

    keywords = [
        "can i get image",
        "show image",
        "display image",
        "get image",
        "show camera",
        "show frame",
        "show me image",
        "camera image",
        "image please",
        "사진",
        "이미지",
        "카메라 보여",
        "화면 보여",
    ]

    return any(k in q for k in keywords)


def get_embedding(text):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }

    res = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json=payload,
        timeout=120
    )
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
        print(f"Vector DB not found: {VECTOR_DB_PATH}")
        return []

    with open(VECTOR_DB_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded vector DB: {VECTOR_DB_PATH}")
    print(f"Total RAG chunks: {len(records)}")

    return records


def retrieve_context(query, records, top_k=TOP_K):
    if not records:
        return ""

    query_embedding = get_embedding(query)

    scored = []

    for record in records:
        embedding = record.get("embedding")
        if embedding is None:
            continue

        score = cosine_similarity(query_embedding, embedding)
        scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)

    context_blocks = []

    for i, (score, record) in enumerate(scored[:top_k], start=1):
        source = record.get("source", "unknown")
        chunk_index = record.get("chunk_index", "unknown")
        text = record.get("text", "")

        context_blocks.append(
            f"""
[RAG_CHUNK_{i}]
score: {score:.4f}
source: {source}
chunk_index: {chunk_index}
content:
{text}
""".strip()
        )

    return "\n\n".join(context_blocks)


def detect_objects(model, frame):
    results = model.predict(
        frame,
        device="cpu",
        verbose=False
    )

    result = results[0]
    object_counts = {}

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id))
            object_counts[cls_name] = object_counts.get(cls_name, 0) + 1

    annotated_frame = result.plot()

    return object_counts, annotated_frame


def format_detected_objects(object_counts):
    if not object_counts:
        return "none"

    lines = []

    for name, count in sorted(object_counts.items()):
        lines.append(f"{name}: {count}")

    return "\n".join(lines)


def build_ontology_facts(object_counts):
    """
    Simple ontology:
    - Type is only person or object.
    - person is type person.
    - every other detected class is type object.
    - If person and work-related objects exist, each person is working.
    """

    facts = []
    types = []
    relations = []
    contexts = []

    person_count = object_counts.get("person", 0)

    work_objects = [
        "laptop",
        "keyboard",
        "mouse",
        "monitor",
        "tv",
        "chair",
        "desk"
    ]

    detected_work_objects = [
        obj for obj in work_objects
        if object_counts.get(obj, 0) > 0
    ]

    for obj, count in sorted(object_counts.items()):
        facts.append(f"{obj}({count})")

        if obj == "person":
            for i in range(1, count + 1):
                types.append(f"person_{i}: person")
        else:
            types.append(f"{obj}: object")

    if person_count > 0 and detected_work_objects:
        for i in range(1, person_count + 1):
            relations.append(f"person_{i} is working")

        contexts.append("office_working_detected")

    elif person_count > 0:
        contexts.append("person_detected")

    else:
        contexts.append("no_person_detected")

    ontology_text = f"""
[FACTS]
{chr(10).join(facts) if facts else "none"}

[TYPES]
{chr(10).join(types) if types else "none"}

[RELATIONS]
{chr(10).join(relations) if relations else "none"}

[CONTEXT]
{chr(10).join(contexts) if contexts else "none"}
""".strip()

    return ontology_text


def direct_answer(question, object_counts, ontology_text):
    """
    Answer simple object-count/existence questions directly.
    This prevents llama3.2:1B from making mistakes on easy facts.
    """

    q = question.lower()

    for obj, count in object_counts.items():
        obj_l = obj.lower()

        if obj_l in q:
            if "how many" in q or "count" in q:
                if obj_l == "person":
                    return f"{count} persons are detected."
                return f"{count} {obj_l}(s) are detected."

            if "is there" in q or "are there" in q:
                if count > 0:
                    return f"Yes, {count} {obj_l}(s) are detected."
                return f"No, {obj_l} is not detected."

    if "working" in q or "work" in q:
        if "is working" in ontology_text:
            person_count = object_counts.get("person", 0)
            return f"Yes, {person_count} persons are working."
        return "Working is not detected."

    return None


def ask_llama(question, detected_text, ontology_text, rag_context):
    prompt = f"""
You are an office situation assistant.

Use this order:
1. YOLO detected objects are sensor facts.
2. Ontology facts are derived facts.
3. RAG context is background knowledge.

Rules:
- Do not invent objects.
- Do not invent people.
- Do not invent windows, desks, bookshelves, meetings, or activities unless provided.
- If ontology says "person_N is working", then you may say that person is working.
- If user asks how many persons, use YOLO person count.
- If user asks whether people are working, use ontology relations.
- Keep answer short and direct.
- Do not output the prompt.
- Do not ask another question.
- If you do not know, say "I do not know."

[YOLO_DETECTED_OBJECTS]
{detected_text}

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
            "num_predict": 80
        }
    }

    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=180
    )
    res.raise_for_status()

    return res.json().get("response", "").strip()


def capture_frame(cap):
    ret, frame = cap.read()

    if not ret:
        return None

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    return frame


def show_image_window(frame, annotated_frame=None):
    if annotated_frame is not None:
        cv2.imshow("YOLO Detection Image", annotated_frame)
    else:
        cv2.imshow("Camera Image", frame)

    print("Image window opened. Press any key inside image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("Loading YOLO model on CPU...")
    yolo = YOLO(YOLO_MODEL_PATH)
    yolo.to("cpu")

    print("Loading RAG vector DB...")
    records = load_vector_db()

    print("Opening RTSP stream...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        return

    print("\nPractice system started.")
    print("RTSP + YOLOv11 CPU + Ontology + RAG + llama3.2:1B")
    print("No live display.")
    print("Type question and press ENTER.")
    print("Example: can i get image")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        frame = capture_frame(cap)

        if frame is None:
            print("Failed to read frame from RTSP camera.")
            continue

        try:
            object_counts, annotated_frame = detect_objects(yolo, frame)
            detected_text = format_detected_objects(object_counts)
            ontology_text = build_ontology_facts(object_counts)

            print("\n[YOLO detected objects]")
            print(detected_text)

            print("\n[ONTOLOGY]")
            print(ontology_text)

            if is_image_request(question):
                show_image_window(frame, annotated_frame)
                print("Assistant: Image displayed.")
                continue

            direct = direct_answer(question, object_counts, ontology_text)

            if direct is not None:
                print("\nAssistant:", direct)
                continue

            rag_query = f"""
Question:
{question}

YOLO detected objects:
{detected_text}

Ontology:
{ontology_text}
"""

            rag_context = retrieve_context(rag_query, records, top_k=TOP_K)

            print("\n[RAG retrieved context]")
            if rag_context:
                print(rag_context[:1200])
                if len(rag_context) > 1200:
                    print("...")
            else:
                print("No RAG context found.")

            answer = ask_llama(
                question=question,
                detected_text=detected_text,
                ontology_text=ontology_text,
                rag_context=rag_context
            )

            print("\nAssistant:", answer)

        except Exception as e:
            print("Error:", e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
