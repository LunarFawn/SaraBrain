/**
 * app.js — Main entry point: load Pyodide, mount sara_brain, wire UI.
 */

import { setPyodide, initBrain, runCommand, getGraphData, getLastRecognitionPaths, exportBrain, importBrain, seedBrain, getQuestionWords, getCandidateProperties, setPerceptionState } from "./bridge.js";
import { saveBrainState, loadBrainState, clearBrainState, downloadBrainExport, uploadBrainImport } from "./persistence.js";
import { initGraph, updateGraph, animateWavefront } from "./graph.js";
import { checkProxyHealth, runPerceptionLoop, generateLabel } from "./vision.js";

// ── State ──

let commandHistory = [];
let historyIndex = -1;
let isProcessing = false;

// Vision state
let visionApiKey = localStorage.getItem("sara_vision_key") || "";
let visionModel = localStorage.getItem("sara_vision_model") || "claude-sonnet-4-20250514";
let proxyUrl = "http://localhost:8765";
let proxyConnected = false;
let proxyPollInterval = null;

// ── DOM refs ──

const loadingScreen = document.getElementById("loading-screen");
const progressBar = document.getElementById("progress-bar");
const loadingStatus = document.getElementById("loading-status");
const app = document.getElementById("app");
const replOutput = document.getElementById("repl-output");
const replInput = document.getElementById("repl-input");
const detailPanel = document.getElementById("detail-panel");

// Vision DOM refs
const visionPanel = document.getElementById("vision-panel");
const proxyDot = document.getElementById("proxy-dot");
const proxyText = document.getElementById("proxy-text");
const proxyInstructions = document.getElementById("proxy-instructions");
const visionControls = document.getElementById("vision-controls");
const visionKeyInput = document.getElementById("vision-key");
const visionModelInput = document.getElementById("vision-model");
const btnPerceive = document.getElementById("btn-perceive");
const codeModal = document.getElementById("code-modal");
const codeModalBody = document.getElementById("code-modal-body");

// ── Loading ──

function setProgress(pct, status) {
  progressBar.style.width = `${pct}%`;
  loadingStatus.textContent = status;
}

async function boot() {
  try {
    setProgress(10, "Loading Pyodide runtime...");
    const pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/",
    });

    setProgress(30, "Loading sqlite3 package...");
    await pyodide.loadPackage("sqlite3");

    setProgress(50, "Mounting sara_brain engine...");
    await mountSaraBrain(pyodide);

    setProgress(60, "Initializing boot.py...");
    // Load and run boot.py
    const bootResponse = await fetch("python/boot.py");
    const bootCode = await bootResponse.text();
    await pyodide.runPythonAsync(bootCode);

    setPyodide(pyodide);

    setProgress(75, "Initializing brain...");
    await initBrain();

    setProgress(85, "Restoring saved state...");
    await restoreSavedState();

    setProgress(95, "Starting UI...");
    initUI();

    setProgress(100, "Ready!");
    await new Promise((r) => setTimeout(r, 300));

    loadingScreen.classList.add("hidden");
    app.classList.add("visible");

    // Initial graph render
    await refreshGraph();

    replInput.focus();
  } catch (err) {
    setProgress(0, `Error: ${err.message}`);
    console.error("Boot failed:", err);
  }
}

async function mountSaraBrain(pyodide) {
  // Fetch the Python source files and mount them into Pyodide's virtual FS
  const baseUrl = getSaraBrainBaseUrl();

  // Create the package directory structure
  const dirs = [
    "/home/pyodide/sara_brain",
    "/home/pyodide/sara_brain/core",
    "/home/pyodide/sara_brain/models",
    "/home/pyodide/sara_brain/parsing",
    "/home/pyodide/sara_brain/repl",
    "/home/pyodide/sara_brain/storage",
    "/home/pyodide/sara_brain/visualization",
  ];

  for (const dir of dirs) {
    try {
      pyodide.FS.mkdirTree(dir);
    } catch {
      // Directory may already exist
    }
  }

  // Map of files to fetch
  const files = [
    "sara_brain/__init__.py",
    "sara_brain/core/__init__.py",
    "sara_brain/core/brain.py",
    "sara_brain/core/learner.py",
    "sara_brain/core/recognizer.py",
    "sara_brain/core/similarity.py",
    "sara_brain/models/__init__.py",
    "sara_brain/models/neuron.py",
    "sara_brain/models/segment.py",
    "sara_brain/models/path.py",
    "sara_brain/models/result.py",
    "sara_brain/parsing/__init__.py",
    "sara_brain/parsing/statement_parser.py",
    "sara_brain/parsing/taxonomy.py",
    "sara_brain/repl/__init__.py",
    "sara_brain/repl/shell.py",
    "sara_brain/repl/commands.py",
    "sara_brain/repl/formatters.py",
    "sara_brain/storage/__init__.py",
    "sara_brain/storage/database.py",
    "sara_brain/storage/neuron_repo.py",
    "sara_brain/storage/segment_repo.py",
    "sara_brain/storage/path_repo.py",
    "sara_brain/storage/association_repo.py",
    "sara_brain/storage/queries.py",
    "sara_brain/storage/schema.sql",
    "sara_brain/visualization/__init__.py",
    "sara_brain/visualization/text_tree.py",
  ];

  // Fetch all files in parallel
  const fetchPromises = files.map(async (filePath) => {
    const url = `${baseUrl}${filePath}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${url}: ${resp.status}`);
    }
    const content = await resp.text();
    return { filePath, content };
  });

  const results = await Promise.all(fetchPromises);

  for (const { filePath, content } of results) {
    pyodide.FS.writeFile(`/home/pyodide/${filePath}`, content);
  }
}

function getSaraBrainBaseUrl() {
  // In gh-pages deployment, the Python files are at python/
  // In local dev, they might be at ../src/
  // We'll try python/ first (gh-pages structure)
  return "python/";
}

async function restoreSavedState() {
  try {
    const saved = await loadBrainState();
    if (saved) {
      const result = await importBrain(saved);
      appendOutput(result, "cmd-output");
      appendOutput("  (Restored from browser storage)", "welcome");
    } else {
      // First visit — seed with demo data
      const result = await seedBrain();
      appendOutput(result, "cmd-output");
    }
  } catch (err) {
    console.warn("Could not restore saved state:", err);
  }
}

// ── UI Initialization ──

function initUI() {
  // Welcome message
  appendOutput("Sara Brain — Path of Thought", "welcome");
  appendOutput('Type "help" for commands. Click a node to trace it.\n', "welcome");

  // Input handling
  replInput.addEventListener("keydown", handleInputKeydown);

  // Button handlers
  document.getElementById("btn-seed").addEventListener("click", handleSeed);
  document.getElementById("btn-reset").addEventListener("click", handleReset);
  document.getElementById("btn-export").addEventListener("click", handleExport);
  document.getElementById("btn-import").addEventListener("click", handleImport);

  // Vision panel handlers
  document.getElementById("btn-vision").addEventListener("click", toggleVisionPanel);
  document.getElementById("btn-download-proxy").addEventListener("click", handleDownloadProxy);
  document.getElementById("btn-view-code").addEventListener("click", handleViewCode);
  document.getElementById("btn-close-modal").addEventListener("click", () => { codeModal.style.display = "none"; });
  btnPerceive.addEventListener("click", handlePerceive);

  // Restore saved vision settings
  if (visionApiKey) visionKeyInput.value = visionApiKey;
  if (visionModel) visionModelInput.value = visionModel;

  visionKeyInput.addEventListener("change", () => {
    visionApiKey = visionKeyInput.value.trim();
    localStorage.setItem("sara_vision_key", visionApiKey);
    updatePerceiveButton();
  });
  visionModelInput.addEventListener("change", () => {
    visionModel = visionModelInput.value.trim();
    localStorage.setItem("sara_vision_model", visionModel);
  });

  // Close modal on backdrop click
  codeModal.addEventListener("click", (e) => {
    if (e.target === codeModal) codeModal.style.display = "none";
  });
}

async function handleInputKeydown(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    const line = replInput.value.trim();
    if (!line || isProcessing) return;

    commandHistory.push(line);
    historyIndex = commandHistory.length;
    replInput.value = "";

    await executeCommand(line);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    if (historyIndex > 0) {
      historyIndex--;
      replInput.value = commandHistory[historyIndex];
    }
  } else if (e.key === "ArrowDown") {
    e.preventDefault();
    if (historyIndex < commandHistory.length - 1) {
      historyIndex++;
      replInput.value = commandHistory[historyIndex];
    } else {
      historyIndex = commandHistory.length;
      replInput.value = "";
    }
  }
}

// ── Command Execution ──

async function executeCommand(line) {
  isProcessing = true;
  appendOutput(`> ${line}`, "cmd-line");

  try {
    const cmd = line.split(/\s+/)[0].toLowerCase();

    // Perceive: trigger file dialog instead of running Python command
    if (cmd === "perceive") {
      await handlePerceive();
      isProcessing = false;
      replInput.focus();
      return;
    }

    // no/see: run command, refresh graph
    if (cmd === "no" || cmd === "see") {
      const output = await runCommand(line);
      appendOutput(output, "cmd-output");
      await refreshGraph();
      await persistState();
      isProcessing = false;
      replInput.focus();
      return;
    }

    // For recognize, run command then get cached path data for animation
    if (cmd === "recognize") {
      const output = await runCommand(line);
      appendOutput(output, "cmd-output");

      // Get cached recognition paths (stored by run_command, no re-run)
      const pathData = await getLastRecognitionPaths();
      await refreshGraph();
      await persistState();

      // Animate wavefront if we got paths
      if (pathData.results && pathData.results.length > 0) {
        const allPaths = pathData.results.flatMap((r) => r.paths);
        if (allPaths.length > 0) {
          animateWavefront(allPaths);
        }
        showDetail(formatRecognitionDetail(pathData));
      }
    } else {
      const output = await runCommand(line);
      appendOutput(output, "cmd-output");

      // Refresh graph after teach, reset, seed, import
      if (["teach", "reset", "seed"].includes(cmd)) {
        await refreshGraph();
        await persistState();
      }

      // Show detail for trace/why
      if (["trace", "why", "similar", "analyze"].includes(cmd)) {
        showDetail(output);
      }
    }
  } catch (err) {
    appendOutput(`Error: ${err.message}`, "cmd-error");
  }

  isProcessing = false;
  replInput.focus();
}

// ── Button Handlers ──

async function handleSeed() {
  await executeCommand("seed");
}

async function handleReset() {
  await executeCommand("reset");
  await clearBrainState();
}

async function handleExport() {
  try {
    const jsonStr = await exportBrain();
    downloadBrainExport(jsonStr);
    appendOutput("  Exported brain to file.", "cmd-output");
  } catch (err) {
    appendOutput(`Export error: ${err.message}`, "cmd-error");
  }
}

async function handleImport() {
  try {
    const jsonStr = await uploadBrainImport();
    const result = await importBrain(jsonStr);
    appendOutput(result, "cmd-output");
    await refreshGraph();
    await persistState();
  } catch (err) {
    if (err.message !== "No file selected") {
      appendOutput(`Import error: ${err.message}`, "cmd-error");
    }
  }
}

// ── Output Helpers ──

function appendOutput(text, className) {
  const div = document.createElement("div");
  div.className = className;
  div.textContent = text;
  replOutput.appendChild(div);
  replOutput.scrollTop = replOutput.scrollHeight;
}

function showDetail(text) {
  detailPanel.innerHTML = "";
  const label = document.createElement("div");
  label.className = "detail-label";
  label.textContent = "Detail";
  detailPanel.appendChild(label);
  const content = document.createElement("div");
  content.textContent = text;
  detailPanel.appendChild(content);
}

function formatRecognitionDetail(pathData) {
  const lines = [];
  for (const r of pathData.results) {
    lines.push(`${r.label} (confidence: ${r.confidence})`);
    for (const path of r.paths) {
      lines.push(`  path: [${path.join(" → ")}]`);
    }
  }
  return lines.join("\n");
}

// ── Graph ──

async function refreshGraph() {
  try {
    const data = await getGraphData();
    if (data.nodes.length > 0) {
      const emptyMsg = document.querySelector(".graph-empty");
      if (emptyMsg) emptyMsg.style.display = "none";
    }
    updateGraph(data, handleNodeClick);
  } catch (err) {
    console.warn("Graph refresh failed:", err);
  }
}

async function handleNodeClick(node) {
  const cmd = node.type === "concept" ? "why" : "trace";
  await executeCommand(`${cmd} ${node.label}`);
}

// ── Persistence ──

async function persistState() {
  try {
    const jsonStr = await exportBrain();
    await saveBrainState(jsonStr);
  } catch (err) {
    console.warn("Failed to persist brain state:", err);
  }
}

// ── Vision Panel ──

function toggleVisionPanel() {
  const visible = visionPanel.style.display !== "none";
  if (visible) {
    visionPanel.style.display = "none";
    stopProxyPolling();
  } else {
    visionPanel.style.display = "";
    startProxyPolling();
  }
}

function startProxyPolling() {
  checkAndUpdateStatus();
  proxyPollInterval = setInterval(checkAndUpdateStatus, 3000);
}

function stopProxyPolling() {
  if (proxyPollInterval) {
    clearInterval(proxyPollInterval);
    proxyPollInterval = null;
  }
}

async function checkAndUpdateStatus() {
  proxyConnected = await checkProxyHealth(proxyUrl);
  if (proxyConnected) {
    proxyDot.classList.add("connected");
    proxyText.textContent = "Proxy: connected";
    proxyInstructions.style.display = "none";
    visionControls.style.display = "";
  } else {
    proxyDot.classList.remove("connected");
    proxyText.textContent = "Proxy: not connected";
    proxyInstructions.style.display = "";
    visionControls.style.display = "none";
  }
  updatePerceiveButton();
}

function updatePerceiveButton() {
  btnPerceive.disabled = !(proxyConnected && visionApiKey);
}

async function handleDownloadProxy() {
  try {
    const resp = await fetch("python/vision_proxy.py");
    const text = await resp.text();
    const blob = new Blob([text], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "vision_proxy.py";
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    appendOutput(`Download error: ${err.message}`, "cmd-error");
  }
}

async function handleViewCode() {
  try {
    const resp = await fetch("python/vision_proxy.py");
    const text = await resp.text();
    codeModalBody.textContent = text;
    codeModal.style.display = "";
  } catch (err) {
    appendOutput(`Error loading source: ${err.message}`, "cmd-error");
  }
}

async function handlePerceive() {
  if (isProcessing) return;
  if (!proxyConnected || !visionApiKey) {
    appendOutput("  Vision not ready. Connect proxy and enter API key.", "cmd-error");
    return;
  }

  // Open file dialog
  const file = await pickImageFile();
  if (!file) return;

  isProcessing = true;
  appendOutput(`> perceive ${file.name}`, "cmd-line");
  appendOutput("  Reading image...", "cmd-output");

  try {
    const arrayBuffer = await file.arrayBuffer();
    const label = await generateLabel(file.name, arrayBuffer);

    // Convert to base64
    const bytes = new Uint8Array(arrayBuffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const imageBase64 = btoa(binary);

    // Determine media type
    const ext = file.name.split(".").pop().toLowerCase();
    const mediaTypes = { jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", gif: "image/gif", webp: "image/webp" };
    const mediaType = mediaTypes[ext] || "image/jpeg";

    appendOutput(`  Label: ${label}`, "cmd-output");
    appendOutput("  Starting perception loop...", "cmd-output");

    const bridge = {
      runCommand,
      getQuestionWords,
      getCandidateProperties,
      setPerceptionState,
    };

    const result = await runPerceptionLoop(
      proxyUrl, visionApiKey, visionModel,
      imageBase64, mediaType, label, bridge,
      (step) => {
        appendOutput(`  [${step.phase}] observed: ${step.observations.join(", ") || "(none)"}`, "perception-obs");
        if (step.recognition) {
          appendOutput(`  ${step.recognition}`, "cmd-output");
        }
      }
    );

    appendOutput(`  Perception complete: ${result.totalTaught} properties taught`, "perception-phase");
    await refreshGraph();
    await persistState();
  } catch (err) {
    appendOutput(`  Perception error: ${err.message}`, "cmd-error");
  }

  isProcessing = false;
  replInput.focus();
}

function pickImageFile() {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/jpeg,image/png,image/gif,image/webp";
    input.onchange = () => {
      resolve(input.files[0] || null);
    };
    // If user cancels, resolve null after a delay
    input.addEventListener("cancel", () => resolve(null));
    input.click();
  });
}

// ── Init Graph & Boot ──

document.addEventListener("DOMContentLoaded", () => {
  initGraph("graph-container");
  boot();
});
