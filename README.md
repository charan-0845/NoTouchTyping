# ASL Fingerspelling Recognition — Deployment

Real-time ASL letter recognition (26 letters + space + delete) using
MediaPipe Hands for landmark extraction and a trained MLP for classification.
Backend: FastAPI + WebSocket. Frontend: plain HTML/JS (webcam capture only —
all ML inference happens server-side).

## Project structure

```
asl_deploy/
├── backend/
│   ├── main.py             # FastAPI app + WebSocket endpoint + model inference
│   ├── requirements.txt
│   ├── Dockerfile
│   └── asl_mlp_model.h5    # <-- YOU NEED TO ADD THIS (your trained model)
└── frontend/
    └── index.html          # webcam capture + display, connects via WebSocket
```

## Step 1 — Add your trained model

Copy your trained `.h5` (or `.keras`) model file into `backend/`, matching
the `MODEL_PATH` variable at the top of `main.py`. Also double check the
`LABELS` list order matches your training label encoding exactly.

## Step 2 — Run locally WITHOUT Docker (fastest way to test)

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` directly in your browser (double-click it,
or use VS Code's Live Server extension). It connects to
`ws://localhost:8000/ws` by default.

You should see your webcam feed, a predicted letter overlay, and the
typed-text box updating as you hold signs steady.

## Step 3 — Run with Docker (this is what actually gets deployed)

Docker packages your backend + all its dependencies (Python, OpenCV,
MediaPipe, TensorFlow, system libraries) into one self-contained image, so
it runs identically on your machine and on a cloud server — no "works on my
machine" surprises.

Install Docker Desktop first: https://www.docker.com/products/docker-desktop/

Then, from the `backend/` folder:

```bash
# Build the image (only needed when code/dependencies change)
docker build -t asl-backend .

# Run a container from that image
docker run -p 8000:8000 asl-backend
```

`-p 8000:8000` maps port 8000 on your machine to port 8000 inside the
container, so `localhost:8000` reaches your app exactly like step 2 did.

Test it the same way — open `frontend/index.html` in your browser.

## Step 4 — Deploy the backend to the cloud

Recommended for this stack (Python + TensorFlow + MediaPipe + WebSockets):
**Render** or **Railway** — both support Docker deployments, persistent
WebSocket connections, and have free/cheap starter tiers. (Avoid serverless
platforms like Vercel/AWS Lambda for this — they don't handle long-lived
WebSocket connections or heavy ML dependencies well.)

General flow (Render as example):
1. Push this `backend/` folder to a GitHub repo.
2. On Render: New → Web Service → connect your repo.
3. Render auto-detects the `Dockerfile` and builds from it.
4. Set the port to `8000` (matches `EXPOSE 8000` in the Dockerfile).
5. Deploy — you'll get a public URL like `https://your-app.onrender.com`.

Note: Render's free tier serves over HTTPS, which means your WebSocket URL
becomes `wss://` (not `ws://`) — update `BACKEND_WS_URL` in `index.html`
accordingly once deployed.

## Step 5 — Deploy the frontend

Since `index.html` is just static HTML/JS, host it for free on:
- **Vercel** or **Netlify** (drag-and-drop the `frontend/` folder, or connect
  the repo)
- **GitHub Pages**

Before deploying, update `BACKEND_WS_URL` in `index.html` to point to your
deployed backend's `wss://` URL instead of `localhost`.

## Notes / known limitations to mention in your portfolio writeup

- Inference runs server-side, so latency depends on your hosting tier's CPU
  (free tiers are slower — consider mentioning this as a tradeoff you made
  and understood, which is itself a good portfolio talking point).
- `normalize_landmarks()` in `main.py` must exactly match whatever
  preprocessing you used during training, or accuracy will look wrong even
  though the model itself is fine.
- CORS is currently wide open (`allow_origins=["*"]`) for ease of local
  testing — tighten this to your actual frontend domain before calling it
  "production-ready."
