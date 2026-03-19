/**
 * imageviewer.js — Canvas-based image viewer with label chips and selection tools.
 *
 * Shows uploaded images, overlays detected labels in real time,
 * and lets users click/drag/lasso to select regions for querying.
 */

// ── Module State ──

let container = null;
let labelsEl = null;
let canvasArea = null;
let imageCanvas = null;
let overlayCanvas = null;
let emptyEl = null;
let toolbarEl = null;
let coordsEl = null;

let imageCtx = null;
let overlayCtx = null;

let fullResImage = null;  // Image object at native resolution
let imageBase64Data = null;
let imageMediaType = null;

let displayRect = { x: 0, y: 0, w: 0, h: 0 }; // letterboxed image rect on canvas

let currentTool = "rect"; // "point" | "rect" | "lasso"
let isDrawing = false;
let drawStart = { x: 0, y: 0 };
let lassoPoints = [];

let regionCallback = null;

// ── Public API ──

export function initImageViewer(containerId) {
  container = document.getElementById(containerId);
  if (!container) return;

  // Build DOM
  container.innerHTML = `
    <div class="iv-labels"></div>
    <div class="iv-canvas-area">
      <canvas id="iv-image-canvas"></canvas>
      <canvas id="iv-overlay-canvas"></canvas>
      <div class="iv-empty">Upload an image to begin</div>
    </div>
    <div class="iv-toolbar">
      <button class="iv-tool-btn active" data-tool="point" title="Point selection (200x200 crop)">Point</button>
      <button class="iv-tool-btn" data-tool="rect" title="Rectangle selection">Rect</button>
      <button class="iv-tool-btn" data-tool="lasso" title="Freeform lasso selection">Lasso</button>
      <span class="iv-coords"></span>
    </div>
  `;

  labelsEl = container.querySelector(".iv-labels");
  canvasArea = container.querySelector(".iv-canvas-area");
  imageCanvas = container.querySelector("#iv-image-canvas");
  overlayCanvas = container.querySelector("#iv-overlay-canvas");
  emptyEl = container.querySelector(".iv-empty");
  toolbarEl = container.querySelector(".iv-toolbar");
  coordsEl = container.querySelector(".iv-coords");

  imageCtx = imageCanvas.getContext("2d");
  overlayCtx = overlayCanvas.getContext("2d");

  // Tool buttons
  for (const btn of toolbarEl.querySelectorAll(".iv-tool-btn")) {
    btn.addEventListener("click", () => {
      currentTool = btn.dataset.tool;
      for (const b of toolbarEl.querySelectorAll(".iv-tool-btn")) {
        b.classList.toggle("active", b === btn);
      }
    });
  }

  // Default to point tool
  currentTool = "point";

  // Pointer events on overlay canvas
  overlayCanvas.addEventListener("pointerdown", handlePointerDown);
  overlayCanvas.addEventListener("pointermove", handlePointerMove);
  overlayCanvas.addEventListener("pointerup", handlePointerUp);
  overlayCanvas.addEventListener("pointerleave", handlePointerUp);

  // ResizeObserver for canvas sizing
  const ro = new ResizeObserver(() => resizeCanvases());
  ro.observe(canvasArea);
}

export function loadImage(base64, mediaType, label) {
  imageBase64Data = base64;
  imageMediaType = mediaType;

  // Clear previous state
  labelsEl.innerHTML = "";
  coordsEl.textContent = "";
  clearOverlay();

  // Create Image object
  fullResImage = new Image();
  fullResImage.onload = () => {
    emptyEl.style.display = "none";
    resizeCanvases();
    drawImage();
  };
  fullResImage.src = `data:${mediaType};base64,${base64}`;
}

export function addLabel(text, phase) {
  const chip = document.createElement("span");
  chip.className = `iv-label iv-label--${phase}`;
  chip.textContent = text;
  labelsEl.appendChild(chip);
  // Trigger fade-in
  requestAnimationFrame(() => chip.classList.add("visible"));
  // Scroll labels to end
  labelsEl.scrollLeft = labelsEl.scrollWidth;
}

export function clearViewer() {
  fullResImage = null;
  imageBase64Data = null;
  imageMediaType = null;
  labelsEl.innerHTML = "";
  coordsEl.textContent = "";
  if (imageCtx) imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  clearOverlay();
  if (emptyEl) emptyEl.style.display = "";
}

export function onRegionSelected(fn) {
  regionCallback = fn;
}

export function hasImage() {
  return fullResImage !== null;
}

// ── Canvas Sizing ──

function resizeCanvases() {
  const rect = canvasArea.getBoundingClientRect();
  const w = Math.floor(rect.width);
  const h = Math.floor(rect.height);

  imageCanvas.width = w;
  imageCanvas.height = h;
  overlayCanvas.width = w;
  overlayCanvas.height = h;

  if (fullResImage) drawImage();
}

function drawImage() {
  if (!fullResImage || !imageCtx) return;

  const cw = imageCanvas.width;
  const ch = imageCanvas.height;
  const iw = fullResImage.naturalWidth;
  const ih = fullResImage.naturalHeight;

  // Letterbox: fit image preserving aspect ratio
  const scale = Math.min(cw / iw, ch / ih);
  const dw = Math.floor(iw * scale);
  const dh = Math.floor(ih * scale);
  const dx = Math.floor((cw - dw) / 2);
  const dy = Math.floor((ch - dh) / 2);

  displayRect = { x: dx, y: dy, w: dw, h: dh };

  imageCtx.clearRect(0, 0, cw, ch);
  imageCtx.drawImage(fullResImage, dx, dy, dw, dh);
}

// ── Coordinate Mapping ──

function canvasToImageCoords(cx, cy) {
  if (!fullResImage) return null;

  const rx = cx - displayRect.x;
  const ry = cy - displayRect.y;

  if (rx < 0 || ry < 0 || rx > displayRect.w || ry > displayRect.h) return null;

  const iw = fullResImage.naturalWidth;
  const ih = fullResImage.naturalHeight;

  const ix = Math.round((rx / displayRect.w) * iw);
  const iy = Math.round((ry / displayRect.h) * ih);

  return { x: Math.max(0, Math.min(ix, iw)), y: Math.max(0, Math.min(iy, ih)) };
}

function getCanvasPos(e) {
  const rect = overlayCanvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

// ── Selection Handling ──

function handlePointerDown(e) {
  if (!fullResImage) return;
  isDrawing = true;
  const pos = getCanvasPos(e);
  drawStart = pos;
  lassoPoints = [pos];
  overlayCanvas.setPointerCapture(e.pointerId);
}

function handlePointerMove(e) {
  if (!isDrawing || !fullResImage) return;
  const pos = getCanvasPos(e);

  clearOverlay();

  if (currentTool === "rect") {
    const x = Math.min(drawStart.x, pos.x);
    const y = Math.min(drawStart.y, pos.y);
    const w = Math.abs(pos.x - drawStart.x);
    const h = Math.abs(pos.y - drawStart.y);
    overlayCtx.fillStyle = "rgba(88, 166, 255, 0.2)";
    overlayCtx.strokeStyle = "rgba(88, 166, 255, 0.8)";
    overlayCtx.lineWidth = 2;
    overlayCtx.fillRect(x, y, w, h);
    overlayCtx.strokeRect(x, y, w, h);
  } else if (currentTool === "lasso") {
    lassoPoints.push(pos);
    overlayCtx.beginPath();
    overlayCtx.moveTo(lassoPoints[0].x, lassoPoints[0].y);
    for (let i = 1; i < lassoPoints.length; i++) {
      overlayCtx.lineTo(lassoPoints[i].x, lassoPoints[i].y);
    }
    overlayCtx.strokeStyle = "rgba(88, 166, 255, 0.8)";
    overlayCtx.lineWidth = 2;
    overlayCtx.stroke();
  }
}

function handlePointerUp(e) {
  if (!isDrawing || !fullResImage) return;
  isDrawing = false;

  const pos = getCanvasPos(e);

  if (currentTool === "point") {
    selectPoint(pos);
  } else if (currentTool === "rect") {
    const dx = Math.abs(pos.x - drawStart.x);
    const dy = Math.abs(pos.y - drawStart.y);
    if (dx < 10 && dy < 10) {
      // Too small — treat as point
      selectPoint(pos);
    } else {
      selectRect(drawStart, pos);
    }
  } else if (currentTool === "lasso") {
    lassoPoints.push(pos);
    if (lassoPoints.length < 3) {
      selectPoint(pos);
    } else {
      selectLasso(lassoPoints);
    }
  }

  lassoPoints = [];
}

function selectPoint(canvasPos) {
  const imgCoord = canvasToImageCoords(canvasPos.x, canvasPos.y);
  if (!imgCoord) return;

  const iw = fullResImage.naturalWidth;
  const ih = fullResImage.naturalHeight;

  // 200x200 crop centered on click, clamped
  const cropW = Math.min(200, iw);
  const cropH = Math.min(200, ih);
  const x = Math.max(0, Math.min(imgCoord.x - Math.floor(cropW / 2), iw - cropW));
  const y = Math.max(0, Math.min(imgCoord.y - Math.floor(cropH / 2), ih - cropH));

  performCrop(x, y, cropW, cropH);
}

function selectRect(start, end) {
  const startImg = canvasToImageCoords(start.x, start.y);
  const endImg = canvasToImageCoords(end.x, end.y);
  if (!startImg || !endImg) return;

  const x = Math.min(startImg.x, endImg.x);
  const y = Math.min(startImg.y, endImg.y);
  const w = Math.abs(endImg.x - startImg.x);
  const h = Math.abs(endImg.y - startImg.y);

  if (w < 1 || h < 1) return;
  performCrop(x, y, w, h);
}

function selectLasso(points) {
  // Convert all points to image coords, find bounding box
  const imgPoints = points.map(p => canvasToImageCoords(p.x, p.y)).filter(Boolean);
  if (imgPoints.length < 2) return;

  const xs = imgPoints.map(p => p.x);
  const ys = imgPoints.map(p => p.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  const w = Math.max(...xs) - x;
  const h = Math.max(...ys) - y;

  if (w < 1 || h < 1) return;
  performCrop(x, y, w, h);
}

function performCrop(x, y, w, h) {
  // Clamp to image bounds
  const iw = fullResImage.naturalWidth;
  const ih = fullResImage.naturalHeight;
  x = Math.max(0, Math.round(x));
  y = Math.max(0, Math.round(y));
  w = Math.round(Math.min(w, iw - x));
  h = Math.round(Math.min(h, ih - y));

  if (w < 1 || h < 1) return;

  // Offscreen canvas crop at native resolution
  const offscreen = document.createElement("canvas");
  offscreen.width = w;
  offscreen.height = h;
  const ctx = offscreen.getContext("2d");
  ctx.drawImage(fullResImage, x, y, w, h, 0, 0, w, h);

  const dataUrl = offscreen.toDataURL("image/png");
  const base64 = dataUrl.replace(/^data:image\/png;base64,/, "");

  // Update coords display
  coordsEl.textContent = `(${x}, ${y}) ${w}x${h}`;

  // Flash feedback on overlay
  flashSelection();

  // Fire callback
  if (regionCallback) {
    regionCallback(base64, "image/png", { x, y, w, h });
  }
}

// ── Visual Feedback ──

function clearOverlay() {
  if (overlayCtx) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }
}

function flashSelection() {
  overlayCanvas.style.border = "2px solid rgba(255, 255, 255, 0.8)";
  setTimeout(() => {
    overlayCanvas.style.border = "";
  }, 200);
}
