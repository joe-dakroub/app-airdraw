// MediaPipe Hands
// https://ai.google.dev/edge/mediapipe/solutions/guide
import "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";
import "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";

/* DOM Elements */
const video = document.getElementById("cam");
const drawCanvas = document.getElementById("drawCanvas");
const overlay = document.getElementById("overlay");
const dCtx = drawCanvas.getContext("2d", {
  alpha: false,
  desynchronized: true,
});
const oCtx = overlay.getContext("2d", {
  alpha: true,
  desynchronized: true,
});
const modeSelect = document.getElementById("mode");
const colorInput = document.getElementById("color");
const sizeInput = document.getElementById("size");
const smoothInput = document.getElementById("smooth");
const opacityInput = document.getElementById("opacity");
const eraserToggle = document.getElementById("eraserToggle");
const randomToggle = document.getElementById("randomToggle");
const clearBtn = document.getElementById("clear");
const saveBtn = document.getElementById("save");
const statusEl = document.getElementById("status");
const sizeOutput = document.getElementById("sizeOutput");
const smoothOutput = document.getElementById("smoothOutput");
const opacityOutput = document.getElementById("opacityOutput");

dCtx.imageSmoothingEnabled = true;
dCtx.imageSmoothingQuality = "high";

/* Slider Background Update */
function updateRangeBackground(input) {
  const min = parseFloat(input.min) || 0;
  const max = parseFloat(input.max) || 100;
  const value = parseFloat(input.value) || 0;
  const percentage = ((value - min) / (max - min)) * 100;

  if (percentage === 0) {
    input.style.background = `transparent`;
  } else if (percentage === 100) {
    input.style.background = `var(--fg)`;
  } else {
    const thumbWidthPx = 3 * 16; // 3rem in pixels
    const trackWidth = input.offsetWidth;
    const thumbRange = trackWidth - thumbWidthPx;
    const thumbCenterPosition =
      thumbWidthPx / 2 + (percentage / 100) * thumbRange;
    const adjustedPercentage = (thumbCenterPosition / trackWidth) * 100;
    input.style.background = `linear-gradient(to right, var(--fg) 0%, var(--fg) ${adjustedPercentage}%, transparent ${adjustedPercentage}%, transparent 100%)`;
  }
}

/* Constants */
const DETECTION_CONFIDENCE = 0.75;

/* State: Cursor & Drawing */
let emaCursor = null;
let emaCursorDisplay = null;
let lastRaw = null;
let lastSmoothed = null;
let velocity = { x: 0, y: 0 };
let drawing = false;
let lostFrames = 0;
const MAX_PREDICT = 6;

/* State: Pinch Detection */
let pinchEMA = null;
const PINCH_ALPHA = 0.3;
const ENTER_PINCH = 0.16 + (1 - DETECTION_CONFIDENCE) * 0.2;
const EXIT_PINCH = ENTER_PINCH + 0.08;
let consBelow = 0;
let consAbove = 0;
const MIN_ENTER = 4;
const MIN_EXIT = 2;
let isPinched = false;
const MAX_START_VEL = 80;

/* State: Index-Extended Detection */
let idxEMA = null;
const IDX_ALPHA = 0.25;
let idxEnter = 0.9;
let idxExit = 0.75;
let idxConsBelow = 0;
let idxConsAbove = 0;
const IDX_ENTER_FRAMES = 3;
const IDX_EXIT_FRAMES = 2;
let indexState = false;

/* Utility Functions */
function beginDot(p) {
  dCtx.save();
  if (isErasing()) {
    dCtx.globalCompositeOperation = "destination-out";
    dCtx.globalAlpha = 1.0;
  } else {
    dCtx.globalCompositeOperation = "source-over";
    dCtx.globalAlpha = getBrushOpacity();
  }
  dCtx.beginPath();
  dCtx.fillStyle = isErasing() ? "rgba(0,0,0,1)" : getBrushColor();
  dCtx.arc(p.x, p.y, Math.max(1, getBrushSize() / 2), 0, Math.PI * 2);
  dCtx.fill();
  dCtx.restore();
}

function beginStroke(x, y) {}

function computeExtensionMetrics(landmarks) {
  const hsize = handSizeNorm(landmarks);
  const wrist = landmarks[0];
  function distToW(ln) {
    return Math.hypot(ln.x - wrist.x, ln.y - wrist.y);
  }
  const idxTip = distToW(landmarks[8]),
    idxPip = distToW(landmarks[6]);
  const midTip = distToW(landmarks[12]),
    midPip = distToW(landmarks[10]);
  const ringTip = distToW(landmarks[16]),
    ringPip = distToW(landmarks[14]);
  const pinkTip = distToW(landmarks[20]),
    pinkPip = distToW(landmarks[18]);
  const thumbNearIdxMcp =
    Math.hypot(
      landmarks[4].x - landmarks[5].x,
      landmarks[4].y - landmarks[5].y
    ) / hsize;
  const idxMetric = (idxTip - idxPip) / hsize;
  const midMetric = (midTip - midPip) / hsize;
  const ringMetric = (ringTip - ringPip) / hsize;
  const pinkMetric = (pinkTip - pinkPip) / hsize;
  return {
    hsize,
    idxMetric,
    midMetric,
    ringMetric,
    pinkMetric,
    thumbNearIdxMcp,
  };
}

function cursorAlphaFromSlider() {
  const v = parseFloat(smoothInput.value);
  return 0.85 - (v / 0.95) * 0.6;
}

function drawCursor(p, ghost = false) {
  oCtx.save();
  oCtx.beginPath();
  if (isErasing() && !ghost) {
    oCtx.fillStyle = "rgba(255,255,255,0.9)";
    oCtx.strokeStyle = "rgba(0,0,0,0.4)";
    oCtx.lineWidth = 2;
  } else if (ghost) {
    oCtx.fillStyle = "rgba(0,0,0,0.12)";
    oCtx.strokeStyle = "rgba(0,0,0,0.15)";
    oCtx.lineWidth = 1;
  } else {
    oCtx.fillStyle = "rgba(0,0,0,0.95)";
    oCtx.strokeStyle = "rgba(0,0,0,0.15)";
    oCtx.lineWidth = 1;
  }
  const r = Math.max(4, getBrushSize() / 2);
  oCtx.arc(p.x, p.y, r, 0, Math.PI * 2);
  oCtx.fill();
  oCtx.stroke();
  oCtx.closePath();
  oCtx.restore();
}

function endStroke() {}

function fitCanvases() {
  [drawCanvas, overlay].forEach((c) => {
    const rect = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  });
}

function getBrushColor() {
  return colorInput.value;
}

function getBrushOpacity() {
  return parseFloat(opacityInput.value);
}

function getBrushSize() {
  return parseFloat(sizeInput.value) * (window.devicePixelRatio || 1);
}

function handSizeNorm(landmarks) {
  const a = landmarks[0],
    b = landmarks[9];
  return Math.hypot(a.x - b.x, a.y - b.y) || 1e-4;
}

function isErasing() {
  return eraserToggle.checked;
}

function lmToCanvas(lm, canvas) {
  const x = (1 - lm.x) * canvas.width;
  const y = lm.y * canvas.height;
  return { x, y };
}

function mpOptionsFromUI() {
  return {
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: DETECTION_CONFIDENCE,
    minTrackingConfidence: Math.max(0.5, DETECTION_CONFIDENCE - 0.2),
  };
}

function predictNext(prev, vel, n) {
  const decay = Math.max(0, 1 - n / (MAX_PREDICT + 1));
  return {
    x: prev.x + vel.x * n * decay,
    y: prev.y + vel.y * n * decay,
  };
}

function setStatus(s) {
  statusEl.textContent = s;
}

function strokeTo(from, to) {
  if (!from) {
    dCtx.beginPath();
    dCtx.moveTo(to.x, to.y);
    dCtx.stroke();
    return;
  }
  dCtx.save();
  if (isErasing()) {
    dCtx.globalCompositeOperation = "destination-out";
    dCtx.globalAlpha = 1.0;
  } else {
    dCtx.globalCompositeOperation = "source-over";
    dCtx.globalAlpha = getBrushOpacity();
  }
  dCtx.lineWidth = getBrushSize();
  dCtx.lineCap = "round";
  dCtx.lineJoin = "round";
  dCtx.strokeStyle = isErasing() ? "rgba(0,0,0,1)" : getBrushColor();
  dCtx.beginPath();
  dCtx.moveTo(from.x, from.y);
  dCtx.lineTo(to.x, to.y);
  dCtx.stroke();
  dCtx.restore();
}

/* Initialize Canvas */
await new Promise((r) => requestAnimationFrame(r));
fitCanvases();

/* MediaPipe Setup */
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions(mpOptionsFromUI());

hands.onResults((results) => {
  fitCanvases();
  oCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    lostFrames++;
    setStatus(
      lostFrames <= MAX_PREDICT
        ? `Hand briefly lost — predicting (${lostFrames})`
        : "No hand detected"
    );
    if (pinchEMA !== null) pinchEMA *= 0.98;
    if (idxEMA !== null) idxEMA *= 0.98;

    if (lostFrames <= MAX_PREDICT && lastSmoothed) {
      const predicted = predictNext(lastSmoothed, velocity, lostFrames);
      if (drawing) strokeTo(lastSmoothed, predicted);
      lastSmoothed = predicted;
      drawCursor(predicted, true);
    } else {
      if (drawing) {
        drawing = false;
        endStroke && endStroke();
      }
    }
    return;
  }

  lostFrames = 0;
  const lm = results.multiHandLandmarks[0];

  const rawCursor = lmToCanvas(lm[8], drawCanvas);
  if (lastRaw) {
    velocity.x = rawCursor.x - lastRaw.x;
    velocity.y = rawCursor.y - lastRaw.y;
  }
  lastRaw = rawCursor;

  const alpha = cursorAlphaFromSlider();
  if (!emaCursor) emaCursor = { x: rawCursor.x, y: rawCursor.y };
  else {
    emaCursor.x = alpha * rawCursor.x + (1 - alpha) * emaCursor.x;
    emaCursor.y = alpha * rawCursor.y + (1 - alpha) * emaCursor.y;
  }
  const smoothed = { x: emaCursor.x, y: emaCursor.y };

  if (modeSelect.value === "pinch") {
    const raw2D = Math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y);
    const rawZ = Math.abs(lm[8].z - lm[4].z);

    const hsize = handSizeNorm(lm);
    const norm2D = raw2D / hsize;
    const normZ = rawZ / hsize;

    const Z_WEIGHT = 0.2;
    let rawPinch = norm2D - Z_WEIGHT * normZ;
    rawPinch = Math.max(0, Math.min(2, rawPinch));

    if (pinchEMA === null) pinchEMA = rawPinch;
    else pinchEMA = PINCH_ALPHA * rawPinch + (1 - PINCH_ALPHA) * pinchEMA;

    if (pinchEMA <= ENTER_PINCH) {
      consBelow++;
      consAbove = 0;
    } else if (pinchEMA >= EXIT_PINCH) {
      consAbove++;
      consBelow = 0;
    } else {
      consBelow = Math.max(0, consBelow - 1);
      consAbove = Math.max(0, consAbove - 1);
    }

    const speed = Math.hypot(velocity.x || 0, velocity.y || 0);

    if (!isPinched && consBelow >= MIN_ENTER && speed < MAX_START_VEL) {
      isPinched = true;
      beginDot(smoothed);
      drawing = true;
    }
    if (isPinched && consAbove >= MIN_EXIT) {
      isPinched = false;
      drawing = false;
    }

    if (isPinched) {
      setStatus("Pinched — drawing");
      strokeTo(lastSmoothed, smoothed);
      drawing = true;
    } else {
      if (drawing) drawing = false;
      setStatus("Hand detected — pinch to draw (or switch to Index mode)");
    }
  } else {
    const m = computeExtensionMetrics(lm);

    let rawExt =
      m.idxMetric * 1.25 -
      m.midMetric * 0.8 -
      m.ringMetric * 0.7 -
      m.pinkMetric * 0.6 +
      m.thumbNearIdxMcp * 0.1;
    rawExt = Math.max(-2, Math.min(2, rawExt));

    if (idxEMA === null) idxEMA = rawExt;
    else idxEMA = IDX_ALPHA * rawExt + (1 - IDX_ALPHA) * idxEMA;

    if (idxEMA >= idxEnter) {
      idxConsBelow++;
      idxConsAbove = 0;
    } else if (idxEMA <= idxExit) {
      idxConsAbove++;
      idxConsBelow = 0;
    } else {
      idxConsBelow = Math.max(0, idxConsBelow - 1);
      idxConsAbove = Math.max(0, idxConsAbove - 1);
    }

    if (!indexState && idxConsBelow >= IDX_ENTER_FRAMES) {
      indexState = true;
      beginDot(smoothed);
      drawing = true;
    }
    if (indexState && idxConsAbove >= IDX_EXIT_FRAMES) {
      indexState = false;
      drawing = false;
    }

    if (indexState) {
      setStatus("Index extended — drawing");
      strokeTo(lastSmoothed, smoothed);
      drawing = true;
    } else {
      if (drawing) drawing = false;
      setStatus("Hand detected — extend index to draw (or switch to Pinch)");
    }
  }

  const displayAlpha = 0.5;
  if (!emaCursorDisplay) {
    emaCursorDisplay = { x: smoothed.x, y: smoothed.y };
  } else {
    emaCursorDisplay.x =
      displayAlpha * smoothed.x + (1 - displayAlpha) * emaCursorDisplay.x;
    emaCursorDisplay.y =
      displayAlpha * smoothed.y + (1 - displayAlpha) * emaCursorDisplay.y;
  }

  drawCursor(
    emaCursorDisplay,
    !(
      (modeSelect.value === "pinch" && isPinched) ||
      (modeSelect.value === "index" && indexState)
    )
  );

  lastSmoothed = smoothed;
});

/* Camera Initialization */
try {
  const camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video, audio: true });
    },
    width: 1280,
    height: 720,
  });
  await camera.start();
  setStatus("Camera started — allow camera and try Pinch or Index modes");
  dCtx.fillStyle = "#ffffff";
  dCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
} catch (err) {
  console.error(err);
  alert("Camera start failed: " + (err.message || err));
  setStatus("Camera error");
}

/* Event Handlers */
window.addEventListener("resize", fitCanvases);

clearBtn.addEventListener("click", () =>
  dCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height)
);

saveBtn.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = `air-draw-${Date.now()}.png`;
  a.href = drawCanvas.toDataURL("image/png");
  a.click();
});

sizeInput.addEventListener("input", (e) => {
  sizeOutput.textContent = Math.round(e.target.value);
  updateRangeBackground(e.target);
});

smoothInput.addEventListener("input", (e) => {
  smoothOutput.textContent = parseFloat(e.target.value).toFixed(2);
  updateRangeBackground(e.target);
});

opacityInput.addEventListener("input", (e) => {
  opacityOutput.textContent = parseFloat(e.target.value).toFixed(2);
  updateRangeBackground(e.target);
});

/* Initialize slider backgrounds */
requestAnimationFrame(() => {
  updateRangeBackground(sizeInput);
  updateRangeBackground(smoothInput);
  updateRangeBackground(opacityInput);
});

/* Random Parameter Changes */
let randomInterval = null;

function randomizeParameters() {
  // Randomly decide which parameter(s) to change
  const choices = Math.random();

  if (choices < 0.25) {
    // Random color
    const r = Math.floor(Math.random() * 256)
      .toString(16)
      .padStart(2, "0");
    const g = Math.floor(Math.random() * 256)
      .toString(16)
      .padStart(2, "0");
    const b = Math.floor(Math.random() * 256)
      .toString(16)
      .padStart(2, "0");
    colorInput.value = `#${r}${g}${b}`;
  } else if (choices < 0.5) {
    // Random size (between 5 and 70)
    const newSize = Math.floor(Math.random() * 66) + 5;
    sizeInput.value = newSize;
    sizeOutput.textContent = newSize;
    updateRangeBackground(sizeInput);
  } else if (choices < 0.75) {
    // Random smooth (between 0.1 and 0.9)
    const newSmooth = (Math.random() * 0.8 + 0.1).toFixed(2);
    smoothInput.value = newSmooth;
    smoothOutput.textContent = newSmooth;
    updateRangeBackground(smoothInput);
  } else {
    // Random opacity (between 0.3 and 1.0)
    const newOpacity = (Math.random() * 0.7 + 0.3).toFixed(2);
    opacityInput.value = newOpacity;
    opacityOutput.textContent = newOpacity;
    updateRangeBackground(opacityInput);
  }
}

randomToggle.addEventListener("change", (e) => {
  if (e.target.checked) {
    // Start randomizing - change every 1-3 seconds
    randomInterval = setInterval(() => {
      randomizeParameters();
    }, Math.random() * 2000 + 1000);
  } else {
    // Stop randomizing
    if (randomInterval) {
      clearInterval(randomInterval);
      randomInterval = null;
    }
  }
});
