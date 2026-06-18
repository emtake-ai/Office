# Office

Sensor-assisted office context understanding system using RTSP camera input, YOLOv11 object detection, face recognition, ontology, RAG, LLaVA, and local Llama models through Ollama.

This repository is designed as a practical Sensor-sLLM office assistant. It receives camera input, extracts visual facts, converts those facts into office ontology context, retrieves office knowledge from embedded documents, and generates natural-language dialogue using a local small language model.

## Repository Purpose

The goal of this project is not only object detection. The goal is to build an office-aware assistant that can answer questions such as:

```text
Is there a person?
How many people are in the office?
Who is there?
What is the current office situation?
What office furniture is detected?
What office utility is detected?
Can I get image?
llava, what do you see?
```

Core pipeline:

```text
RTSP Camera
  ↓
YOLOv11 Object Detection
  ↓
Face Recognition
  ↓
Ontology
  ↓
RAG with nomic-embed-text
  ↓
Llama3.2:3B / Llama3.2:1B
  ↓
Console Dialogue
```

For image captioning questions, the system can also route the current camera frame to LLaVA.

## Main Features

### RTSP Camera Input

The system receives live camera frames from an RTSP stream.

```python
RTSP_URL = "rtsp://admin:password@192.168.1.7:554/Streaming/Channels/101"
```

The camera is not always displayed live. Instead, the latest frame is kept in memory and used when a user asks a question.

### YOLOv11 Object Detection

YOLOv11 is used to detect office objects such as:

```text
person
chair
laptop
keyboard
mouse
monitor
tv
desk
camera
microphone
speaker
```

Example sensor facts:

```text
person: 2
chair: 2
laptop: 1
keyboard: 1
tv: 3
```

For simple questions, Python answers directly from YOLO facts instead of asking the LLM.

```text
You: how many person is there?
Assistant: 2 person(s) are detected.
```

### Face Recognition

The system uses InsightFace `FaceAnalysis` with known face embeddings stored in:

```text
known_faces/
```

Known faces are stored as `.npy` embedding files.

```text
known_faces/
  kevin.npy
```

The system compares detected face embeddings with known embeddings and returns names such as:

```text
Kevin is there.
Unknown person is there.
```

When the user asks:

```text
You: may i know who is there?
```

the system can show cropped face images in OpenCV windows without saving them to disk.

### Office Ontology

The ontology converts raw detection into meaning.

Raw sensor facts:

```text
person(2)
chair(2)
laptop(1)
keyboard(1)
```

Ontology types:

```text
person_1: person
person_2: person
chair: object
laptop: object
keyboard: object
```

Ontology groups:

```text
furniture: chair, desk, table
utility: laptop, keyboard, mouse, monitor, tv, camera, microphone, speaker
```

Ontology contexts:

```text
office_occupied
office_furniture_detected
office_utility_detected
office_working_detected
office_environment_detected
```

Example rule:

```text
IF person > 0
AND laptop OR keyboard OR monitor exists
THEN office_working_detected
```

The LLM does not invent the office situation. It receives derived context from ontology.

### RAG with nomic-embed-text

Documents in `doc/*.pdf` can be embedded using Ollama `nomic-embed-text`.

Expected vector DB:

```text
embedding/office_embeddings.json
```

Pipeline:

```text
doc/*.pdf
  ↓
text extraction
  ↓
chunking
  ↓
nomic-embed-text
  ↓
office_embeddings.json
```

The runtime then retrieves relevant office knowledge for user questions.

### Llama3.2 Dialogue

The main dialogue model is local through Ollama:

```text
llama3.2:3b
```

Optional smaller model:

```text
llama3.2:1B
```

The LLM receives:

```text
YOLO detected objects
Face recognition result
Ontology context
RAG retrieved context
User question
```

Then it generates a short natural-language answer.

### LLaVA Image Description Mode

If the user starts the question with:

```text
llava,
```

the system routes the current camera frame to:

```text
llava:latest
```

Example:

```text
You: llava, what do you see?
Assistant: The image shows an office workspace with people and computers...
```

Important: LLaVA can hallucinate. For object counts and office facts, YOLO + ontology should be trusted more than LLaVA.

## Repository Structure

```text
Office/
├── basic/
├── doc/
├── embedding/
│   └── office_embeddings.json
├── known_faces/
│   └── *.npy
├── llava/
├── face.py
├── office.py
├── office_best_previous.py
├── office_previous.py
├── office_retina.py
├── practice.py
├── practice_previous.py
├── practice_yolo.py
├── recognize_face.py
├── yolo11n.pt
└── README.md
```

## Main Runtime

```bash
python3 office.py
```

It supports:

```text
RTSP camera
YOLOv11 object detection
InsightFace face recognition
Office ontology
RAG with nomic-embed-text
Llama3.2:3B dialogue
LLaVA image question mode
OpenCV image display on demand
```

## Installation

```bash
pip install ultralytics opencv-python requests numpy insightface onnxruntime pymupdf
```

If you use CPU only, `onnxruntime` is enough.

## Ollama Models

Install required local models:

```bash
ollama pull llama3.2:3b
ollama pull llama3.2:1B
ollama pull nomic-embed-text
ollama pull llava:latest
```

Check installed models:

```bash
ollama list
```

## Running Ollama

Start Ollama manually:

```bash
ollama serve
```

Or use systemd:

```bash
sudo systemctl start ollama
sudo systemctl enable ollama
```

Check server:

```bash
curl http://localhost:11434/api/tags
```

## Running the Office Assistant

```bash
cd ~/Work/Git/Office
python3 office.py
```

Example console:

```text
Assistant: Practice system started. Type a question. Type 'exit' to stop.
Assistant: Use 'llava, your question' to ask LLaVA about the current image.

You: how many person is there?
Assistant: 2 person(s) are detected.

You: may i know who is there?
Assistant: Kevin is there.

You: Let me know on office environment
Assistant: Office situation: 2 person(s) detected; names: Kevin; furniture: 2 chair(s); utility: 1 laptop(s), 1 keyboard(s); office work detected.

You: llava, can you describe the scene?
Assistant: The image shows an office workspace...
```

## Image Display

The system does not display a live camera window by default.

To show current camera image:

```text
You: can i get image
```

To show recognized face crop:

```text
You: may i know who is there?
```

Images are kept in memory and are not saved to disk.

## Person Appearance Monitoring

The system can monitor person appearance at a fixed interval.

```python
MONITOR_INTERVAL = 1.0
```

This means one frame object detection per second.

When a person appears:

```text
Assistant: new person is there
```

When a person disappears:

```text
Assistant: person disappeared
```

## Why Ontology Is Needed

YOLO gives facts:

```text
person: 2
laptop: 1
keyboard: 1
chair: 2
```

But YOLO does not know meaning.

Ontology converts facts into context:

```text
office_occupied
office_furniture_detected
office_utility_detected
office_working_detected
```

Then the LLM can answer more reliably:

```text
The office is occupied, and office work is detected.
```

Without ontology, a small LLM may hallucinate or fail to answer simple structured questions.

## Suggested Office Ontology

### Types

```text
person
object
```

### Object Groups

```text
furniture:
  chair
  desk
  table
  bench
  couch

utility:
  laptop
  keyboard
  mouse
  monitor
  tv
  camera
  microphone
  speaker
  printer
  cell phone
```

### Context Rules

```text
IF person > 0
THEN office_occupied

IF chair OR desk OR table
THEN office_furniture_detected

IF laptop OR keyboard OR mouse OR monitor OR tv
THEN office_utility_detected

IF person > 0
AND laptop OR keyboard OR monitor
THEN office_working_detected

IF person == 0
THEN office_empty
```

### Relation Examples

```text
person_1 identity is Kevin
chair is office furniture
laptop is office utility
Kevin is working
```

## RAG Document Embedding

Put PDF documents in:

```text
doc/
```

Build embeddings into:

```text
embedding/office_embeddings.json
```

The RAG database is used as background knowledge for office-related dialogue.

Example document content should be written in chunk-friendly form:

```text
[SCENARIO]
office_work

[OBJECTS]
person
chair
laptop
keyboard

[RULE]
IF person AND laptop AND keyboard
THEN office_work_possible

[ANSWER]
Office work is possible.
```

## Recommended Architecture

```text
RTSP Camera
  ↓
YOLOv11
  ↓
Object Facts
  ↓
InsightFace
  ↓
Face Identity
  ↓
Office Ontology
  ↓
Derived Context
  ↓
RAG
  ↓
Llama3.2:3B
  ↓
Dialogue
```

Optional:

```text
User input starts with "llava,"
  ↓
Current Frame
  ↓
LLaVA
  ↓
Image Caption Answer
```

## Example Dialogue

```text
You: is there a person?
Assistant: Yes, 2 person(s) are detected.

You: how many chair can you see?
Assistant: 2 chair(s) are detected.

You: Let me know on office furniture
Assistant: Office furniture: 2 chair(s).

You: Let me know on office utility
Assistant: Office utility: 1 laptop(s), 1 keyboard(s), 3 tv(s).

You: what are people doing?
Assistant: Office work is detected.

You: llava, what do you see?
Assistant: The image shows an office scene...
```

## Notes and Limitations

- YOLO object count is treated as sensor fact.
- LLaVA is used only for image captioning and may hallucinate.
- Face recognition depends on the quality of `known_faces/*.npy`.
- Ontology rules are weak if spatial relations are not available.
- “working” is inferred from detected person plus office work objects, not from actual hand or pose tracking.
- For stronger activity recognition, add pose estimation, tracking, or spatial relation detection.
- RTSP warnings such as `bad cseq` can happen due to network packet loss.
- CPU mode is safer on older GPUs.

## Future Work

Recommended improvements:

```text
1. Add person tracking ID
2. Add pose recognition
3. Add spatial relation estimation
4. Add event memory
5. Add daily office activity summary
6. Add face registration script
7. Add better office-specific YOLO training
8. Add Korean dialogue support
9. Add persistent vector memory
10. Add systemd service for office.py
```

## Author

emtake-ai / Office

Sensor-sLLM office assistant project.
