import os
import glob
import json
import hashlib
import requests
import fitz  # PyMuPDF
import numpy as np

DOC_DIR = "./../doc"
DB_DIR = "./"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def read_pdf_text(pdf_path):
    text = ""

    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text.strip():
            text += f"\n[PAGE {page_num}]\n{page_text}\n"

    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_embedding(text):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }

    res = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
    res.raise_for_status()

    return res.json()["embedding"]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_vector_db():
    os.makedirs(DB_DIR, exist_ok=True)

    pdf_files = glob.glob(os.path.join(DOC_DIR, "*.pdf"))

    if not pdf_files:
        print("No PDF files found in ./doc")
        return

    records = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path}")

        text = read_pdf_text(pdf_path)
        chunks = chunk_text(text)

        print(f"Chunks: {len(chunks)}")

        for idx, chunk in enumerate(chunks):
            print(f"Embedding chunk {idx + 1}/{len(chunks)}")

            embedding = get_embedding(chunk)

            record = {
                "id": make_id(pdf_path + str(idx) + chunk),
                "source": pdf_path,
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding
            }

            records.append(record)

    output_path = os.path.join(DB_DIR, "office_embeddings.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved vector DB: {output_path}")
    print(f"Total chunks: {len(records)}")


def search_vector_db(query, top_k=3):
    db_path = os.path.join(DB_DIR, "office_embeddings.json")

    with open(db_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    query_embedding = get_embedding(query)

    results = []

    for record in records:
        score = cosine_similarity(query_embedding, record["embedding"])
        results.append((score, record))

    results.sort(key=lambda x: x[0], reverse=True)

    return results[:top_k]


def main():
    build_vector_db()

    print("\nRAG search ready.")
    print("Type question. Type 'exit' to quit.")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        results = search_vector_db(query, top_k=3)

        print("\nTop retrieved chunks:")

        for score, record in results:
            print("\n==============================")
            print("Score:", score)
            print("Source:", record["source"])
            print("Chunk:", record["chunk_index"])
            print(record["text"][:1000])


if __name__ == "__main__":
    main()
