/**
 * guided.js — Guided mode: action buttons, inline flows, hover hints.
 *
 * Replaces the REPL-first experience with action buttons that guide users
 * through teach/recognize/perceive/explore. A toggle switches between
 * Guided mode and REPL mode — both share the same output log.
 */

// ── Module State ──

let containerEl = null;
let callbacks = {};
let actionsEl = null;
let flowEl = null;
let perceptionFollowUps = null;
let currentFlow = null;

// Flow definitions
const FLOWS = {
  teach: {
    title: "Teach Sara",
    placeholder: "apples are red",
    hint: "Format: subject are property",
    submitLabel: "Teach",
    callback: "onTeach",
  },
  recognize: {
    title: "Recognize",
    placeholder: "red, round, sweet",
    hint: "Comma-separated properties",
    submitLabel: "Recognize",
    callback: "onRecognize",
  },
  explore: {
    title: "Explore",
    placeholder: "apples",
    hint: "Neuron or concept name",
    submitLabel: null, // Has sub-buttons instead
    callback: null,
  },
  correct: {
    title: "Correct Sara",
    placeholder: "ball",
    hint: "What it actually is",
    submitLabel: "Correct",
    callback: "onCorrect",
  },
  pointout: {
    title: "Point Out",
    placeholder: "seams",
    hint: "Property Sara missed",
    submitLabel: "Teach",
    callback: "onSee",
  },
};

// ── Public API ──

export function initGuided(el, cbs) {
  containerEl = el;
  callbacks = cbs;

  containerEl.innerHTML = `
    <div class="guided-actions" id="guided-actions">
      <div class="guided-btn-row">
        <button class="guided-btn" data-flow="teach" title="Teach Sara a new fact">
          Teach
          <span class="guided-btn-hint">Teach Sara a new fact</span>
        </button>
        <button class="guided-btn" data-flow="recognize" title="What could these properties be?">
          Recognize
          <span class="guided-btn-hint">What could these properties be?</span>
        </button>
        <button class="guided-btn" data-action="perceive" title="Show Sara an image to identify">
          Perceive
          <span class="guided-btn-hint">Show Sara an image to identify</span>
        </button>
        <button class="guided-btn" data-flow="explore" title="Look up what Sara knows">
          Explore
          <span class="guided-btn-hint">Look up what Sara knows</span>
        </button>
      </div>
      <div class="guided-btn-row guided-followups" id="guided-followups" style="display:none">
        <button class="guided-btn guided-btn--followup" data-flow="correct" title="That's wrong — tell Sara what it really is">
          Correct
          <span class="guided-btn-hint">That's wrong — tell Sara what it really is</span>
        </button>
        <button class="guided-btn guided-btn--followup" data-flow="pointout" title="Sara missed something — point it out">
          Point Out
          <span class="guided-btn-hint">Sara missed something — point it out</span>
        </button>
      </div>
    </div>
    <div class="guided-flow" id="guided-flow" style="display:none">
      <div class="guided-flow-header">
        <span class="guided-flow-title"></span>
        <button class="btn guided-flow-close">Cancel</button>
      </div>
      <div class="guided-flow-body">
        <input type="text" class="guided-flow-input" autocomplete="off" spellcheck="false" />
        <div class="guided-flow-hint"></div>
      </div>
      <div class="guided-flow-actions"></div>
      <div class="guided-flow-progress" style="display:none">
        <span class="guided-flow-status">Working...</span>
        <button class="btn guided-flow-cancel">Cancel</button>
      </div>
    </div>
  `;

  actionsEl = containerEl.querySelector("#guided-actions");
  flowEl = containerEl.querySelector("#guided-flow");
  perceptionFollowUps = containerEl.querySelector("#guided-followups");

  // Bind action buttons
  for (const btn of containerEl.querySelectorAll(".guided-btn[data-flow]")) {
    btn.addEventListener("click", () => showFlow(btn.dataset.flow));
  }

  // Perceive button (no flow — triggers file dialog directly)
  const perceiveBtn = containerEl.querySelector('[data-action="perceive"]');
  if (perceiveBtn) {
    perceiveBtn.addEventListener("click", () => {
      if (callbacks.onPerceive) callbacks.onPerceive();
    });
  }

  // Cancel buttons
  flowEl.querySelector(".guided-flow-close").addEventListener("click", () => hideFlow());
  flowEl.querySelector(".guided-flow-cancel").addEventListener("click", () => hideFlow());
}

export function showFlow(flowName) {
  const def = FLOWS[flowName];
  if (!def) return;

  currentFlow = flowName;

  // Populate flow panel
  flowEl.querySelector(".guided-flow-title").textContent = def.title;
  const input = flowEl.querySelector(".guided-flow-input");
  input.value = "";
  input.placeholder = def.placeholder;
  flowEl.querySelector(".guided-flow-hint").textContent = def.hint;

  // Build action buttons
  const actionsArea = flowEl.querySelector(".guided-flow-actions");
  actionsArea.innerHTML = "";

  if (flowName === "explore") {
    // Two sub-buttons: Why and Trace
    const whyBtn = document.createElement("button");
    whyBtn.className = "btn btn-accent";
    whyBtn.textContent = "Why";
    whyBtn.addEventListener("click", () => submitFlow("onWhy"));

    const traceBtn = document.createElement("button");
    traceBtn.className = "btn btn-accent";
    traceBtn.textContent = "Trace";
    traceBtn.addEventListener("click", () => submitFlow("onTrace"));

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "btn";
    cancelBtn.textContent = "Cancel";
    cancelBtn.addEventListener("click", () => hideFlow());

    actionsArea.appendChild(whyBtn);
    actionsArea.appendChild(traceBtn);
    actionsArea.appendChild(cancelBtn);
  } else {
    const submitBtn = document.createElement("button");
    submitBtn.className = "btn btn-accent";
    submitBtn.textContent = def.submitLabel;
    submitBtn.addEventListener("click", () => submitFlow(def.callback));

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "btn";
    cancelBtn.textContent = "Cancel";
    cancelBtn.addEventListener("click", () => hideFlow());

    actionsArea.appendChild(submitBtn);
    actionsArea.appendChild(cancelBtn);
  }

  // Show flow, hide progress
  flowEl.style.display = "";
  flowEl.querySelector(".guided-flow-body").style.display = "";
  flowEl.querySelector(".guided-flow-actions").style.display = "";
  flowEl.querySelector(".guided-flow-progress").style.display = "none";

  // Focus input
  setTimeout(() => input.focus(), 50);

  // Enter key submits
  input.onkeydown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (flowName === "explore") {
        submitFlow("onWhy"); // Default to Why on enter
      } else if (def.callback) {
        submitFlow(def.callback);
      }
    }
  };
}

export function hideFlow() {
  flowEl.style.display = "none";
  currentFlow = null;
}

export function setMode(mode) {
  if (containerEl) {
    containerEl.style.display = mode === "guided" ? "" : "none";
  }
}

export function showPerceptionFollowUps(show) {
  if (perceptionFollowUps) {
    perceptionFollowUps.style.display = show ? "" : "none";
  }
}

// ── Internal ──

async function submitFlow(callbackName) {
  const input = flowEl.querySelector(".guided-flow-input");
  const value = input.value.trim();
  if (!value) return;

  const cb = callbacks[callbackName];
  if (!cb) return;

  // Show progress
  flowEl.querySelector(".guided-flow-body").style.display = "none";
  flowEl.querySelector(".guided-flow-actions").style.display = "none";
  flowEl.querySelector(".guided-flow-progress").style.display = "";

  try {
    await cb(value);
  } catch (err) {
    console.warn("Guided flow error:", err);
  }

  // Auto-close flow
  hideFlow();
}
