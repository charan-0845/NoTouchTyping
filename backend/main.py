"""
FastAPI backend for real-time ASL fingerspelling prediction.

Architecture:
    Browser webcam --(JPEG frame over WebSocket)--> FastAPI
    FastAPI: decode frame -> MediaPipe Hands -> normalize landmarks
             -> MLP model -> predicted letter -> send back over WebSocket

Run locally for testing:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open frontend/index.html in a browser (it connects to ws://localhost:8000/ws).
"""

import base64
import io
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

from autocorrect import correct_last_word

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_PATH = "asl_mlp_model.keras"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [ "del","space" ]  # must match the order used during training
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 8

# ---------------------------------------------------------------------------
# App + model setup (loaded once at startup, shared across connections)
# ---------------------------------------------------------------------------
app = FastAPI(title="ASL Fingerspelling API")

# Allow your frontend (served from anywhere during dev) to connect.
# Tighten this to your actual deployed frontend domain before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://charan-0845.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model(MODEL_PATH)
mp_hands = mp.solutions.hands


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    """Must match training preprocessing exactly. See note in earlier script."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )
    wrist = coords[0].copy()
    coords -= wrist
    ref_dist = np.linalg.norm(coords[9])
    if ref_dist > 1e-6:
        coords /= ref_dist
    return coords.flatten()


def predict_letter(landmarks_vector: np.ndarray):
    x = landmarks_vector.reshape(1, -1).astype(np.float32)
    # model.predict() has per-call overhead (dataset/pipeline setup) that adds
    # noticeable latency when called repeatedly on single frames. Calling the
    # model directly avoids that overhead and is much faster for this use case.
    probs = model(x, training=False).numpy()[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])


def decode_base64_image(data_url: str) -> np.ndarray:
    """Frontend sends 'data:image/jpeg;base64,<...>' — strip header, decode to BGR image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


@app.get("/")
def health_check():
    return {"status": "ok", "message": "ASL prediction API is running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    recent_preds = deque(maxlen=SMOOTHING_WINDOW)
    typed_text = ""
    last_committed = None

    # One Hands() instance per connection keeps state isolated between users
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,  # lighter/faster model — trades some accuracy for speed
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            data_url = await websocket.receive_text()

            if data_url == "__reset__":
                typed_text = ""
                last_committed = None
                recent_preds.clear()
                await websocket.send_json(
                    {
                        "prediction": None,
                        "confidence": 0.0,
                        "stable_label": None,
                        "committed": None,
                        "typed_text": typed_text,
                        "landmarks": None,
                        "finished_word": None,
                        "was_corrected": False,
                    }
                )
                continue

            frame = decode_base64_image(data_url)
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label, conf = None, 0.0

            landmarks_out = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                feats = normalize_landmarks(hand_landmarks)
                label, conf = predict_letter(feats)
                recent_preds.append(label if conf >= CONFIDENCE_THRESHOLD else "uncertain")
                # Send normalized (0-1) x,y for each of the 21 points so the
                # frontend can draw them regardless of its own canvas size.
                landmarks_out = [
                    {"x": lm.x, "y": lm.y} for lm in hand_landmarks.landmark
                ]
            else:
                recent_preds.append("none")

            stable_label, count = (
                Counter(recent_preds).most_common(1)[0] if recent_preds else (None, 0)
            )

            committed_this_frame = None
            finished_word = None
            was_corrected = False
            if (
                stable_label not in ("uncertain", "none", None)
                and count >= int(SMOOTHING_WINDOW * 0.7)
                and stable_label != last_committed
            ):
                if stable_label == "space":
                    typed_text, finished_word, was_corrected = correct_last_word(typed_text)
                elif stable_label == "del":
                    typed_text = typed_text[:-1]
                else:
                    typed_text += stable_label
                last_committed = stable_label
                committed_this_frame = stable_label
            elif count < int(SMOOTHING_WINDOW * 0.7):
                last_committed = None

            await websocket.send_json(
                {
                    "prediction": label,
                    "confidence": round(conf, 3),
                    "stable_label": stable_label,
                    "committed": committed_this_frame,
                    "typed_text": typed_text,
                    "landmarks": landmarks_out,
                    "finished_word": finished_word,
                    "was_corrected": was_corrected,
                }
            )

    except WebSocketDisconnect:
        hands.close()
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        hands.close()