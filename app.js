const BACKEND_WS_URL = "wss://notouchtyping.onrender.com/ws"; // change to your deployed backend URL later

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const letterEl = document.getElementById("letter");
const confEl = document.getElementById("conf");
const typedTextEl = document.getElementById("typedText");
const statusEl = document.getElementById("status");
const statusDot = document.getElementById("statusDot");
const clearBtn = document.getElementById("clearBtn");
const audioBtn = document.getElementById("audioBtn");
const landmarkCanvas = document.getElementById("landmarkOverlay");
const landmarkCtx = landmarkCanvas.getContext("2d");

let audioEnabled = true;

// Standard MediaPipe Hands connections (pairs of landmark indices to draw lines between)
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],          // thumb
  [0,5],[5,6],[6,7],[7,8],          // index
  [5,9],[9,10],[10,11],[11,12],     // middle
  [9,13],[13,14],[14,15],[15,16],   // ring
  [13,17],[17,18],[18,19],[19,20],  // pinky
  [0,17]                            // palm base
];

function drawLandmarks(landmarks) {
  landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
  if (!landmarks) return;

  const w = landmarkCanvas.width;
  const h = landmarkCanvas.height;

  landmarkCtx.strokeStyle = "#3D7FFF";
  landmarkCtx.lineWidth = 2;
  HAND_CONNECTIONS.forEach(([a, b]) => {
    const p1 = landmarks[a], p2 = landmarks[b];
    landmarkCtx.beginPath();
    landmarkCtx.moveTo(p1.x * w, p1.y * h);
    landmarkCtx.lineTo(p2.x * w, p2.y * h);
    landmarkCtx.stroke();
  });

  landmarkCtx.fillStyle = "#34D399";
  landmarks.forEach(p => {
    landmarkCtx.beginPath();
    landmarkCtx.arc(p.x * w, p.y * h, 4, 0, 2 * Math.PI);
    landmarkCtx.fill();
  });
}

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.1;
  utterance.volume = 1;
  window.speechSynthesis.speak(utterance);
}

let ws;
let sending = false;

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  video.srcObject = stream;
  return new Promise(resolve => { video.onloadedmetadata = () => resolve(); });
}

function connectWebSocket() {
  ws = new WebSocket(BACKEND_WS_URL);

  ws.onopen = () => {
    statusEl.textContent = "Connected — show a hand sign to the camera";
    statusDot.classList.add("live");
    startStreaming();
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.error) {
      statusEl.textContent = "Backend error: " + data.error;
      statusDot.classList.remove("live");
      return;
    }

    letterEl.textContent = data.prediction ? data.prediction : "…";
    confEl.textContent = data.confidence ? `(${(data.confidence * 100).toFixed(0)}%)` : "";

    if (data.typed_text) {
      typedTextEl.textContent = data.typed_text;
      typedTextEl.classList.remove("empty");
    } else {
      typedTextEl.textContent = "Typed text will appear here…";
      typedTextEl.classList.add("empty");
    }

    drawLandmarks(data.landmarks);

    if (data.finished_word) {
      statusEl.textContent = data.was_corrected
        ? `Autocorrected to "${data.finished_word}"`
        : `Word: "${data.finished_word}"`;
      if (audioEnabled) speak(data.finished_word);
    } else if (data.committed && audioEnabled) {
      const spoken = data.committed === "space" ? "space"
                    : data.committed === "del" ? "delete"
                    : data.committed;
      speak(spoken);
    }

    sending = true;
    sendLoop();
  };

  ws.onclose = () => {
    statusEl.textContent = "Disconnected — retrying in 2s…";
    statusDot.classList.remove("live");
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => {
    statusEl.textContent = "Connection error";
    statusDot.classList.remove("live");
  };
}

function startStreaming() {
  const scale = Math.min(1, 360 / (video.videoWidth || 640));
  canvas.width = Math.round((video.videoWidth || 640) * scale);
  canvas.height = Math.round((video.videoHeight || 480) * scale);

  landmarkCanvas.width = video.clientWidth;
  landmarkCanvas.height = video.clientHeight;

  sending = true;
  sendLoop();
}

function sendLoop() {
  if (ws.readyState !== WebSocket.OPEN) {
    setTimeout(sendLoop, 100);
    return;
  }
  if (!sending) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
  ws.send(dataUrl);
  sending = false;
}

clearBtn.onclick = () => {
  typedTextEl.textContent = "Typed text will appear here…";
  typedTextEl.classList.add("empty");
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("__reset__");
  }
};

audioBtn.onclick = () => {
  audioEnabled = !audioEnabled;
  audioBtn.textContent = audioEnabled ? "🔊 Audio: On" : "🔇 Audio: Off";
  if (!audioEnabled) window.speechSynthesis.cancel();
};

(async function init() {
  await startCamera();
  connectWebSocket();
})();
