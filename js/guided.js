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
    placeholder: "e.g., apples are red",
    hint: "Teach a fact. Format: <subject> is <property> or <subject> is a <concept>",
    submitLabel: "Teach",
    callback: "onTeach",
  },
  recognize: {
    title: "Recognize",
    placeholder: "e.g., red, round, sweet",
    hint: "Find concepts with these properties. Comma-separated.",
    submitLabel: "Recognize",
    callback: "onRecognize",
  },
  explore: {
    title: "Explore",
    placeholder: "e.g., apples",
    hint: "Look up a concept or neuron by name.",
    submitLabel: null, // Has sub-buttons instead
    callback: null,
  },
  associations: {
    title: "List Associations",
    placeholder: "",
    hint: "Show all defined association types.",
    submitLabel: "List",
    callback: "onAssociations",
  },
  define: {
    title: "Define Association",
    placeholder: "e.g., fruit",
    hint: "Define a new type of association.",
    submitLabel: "Define",
    callback: "onDefine",
  },
  describe: {
    title: "Describe Association",
    placeholder: "e.g., fruit as color, shape, taste",
    hint: "Describe an association with properties. Format: <assoc> as <prop1>, <prop2>",
    submitLabel: "Describe",
    callback: "onDescribe",
  },
  similar: {
    title: "Find Similar",
    placeholder: "e.g., apple",
    hint: "Find neurons similar to this one.",
    submitLabel: "Find",
    callback: "onSimilar",
  },
  analyze: {
    title: "Analyze Similarity",
    placeholder: "",
    hint: "Run a full similarity analysis across all neurons.",
    submitLabel: "Analyze",
    callback: "onAnalyze",
  },
  correct: {
    title: "Correct Sara",
    placeholder: "e.g., ball",
    hint: "If Sara recognized something incorrectly, provide the correct label.",
    submitLabel: "Correct",
    callback: "onCorrect",
  },
  pointout: {
    title: "Point Out",
    placeholder: "e.g., seams",
    hint: "If Sara perceived an image and missed a property, point it out.",
    submitLabel: "Teach",
    callback: "onSee",
  },
  neurons: {
    title: "List Neurons",
    placeholder: "",
    hint: "Show all neurons in the brain.",
    submitLabel: "List",
    callback: "onNeurons",
  },
  paths: {
    title: "List Paths",
    placeholder: "",
    hint: "Show all concept paths in the brain.",
    submitLabel: "List",
    callback: "onPaths",
  },
  stats: {
    title: "Show Stats",
    placeholder: "",
    hint: "Show brain statistics.",
    submitLabel: "Show",
    callback: "onStats",
  },
};

// ── Public API ──

export function initGuided(el, cbs) {
  containerEl = el;
  callbacks = cbs;

  containerEl.innerHTML = `
    <div class="guided-actions" id="guided-actions">
      <div class="guided-btn-row">
        <button class="guided-btn" data-flow="teach" title="Teach a fact. e.g., 'apples are red'">
          Teach
          <span class="guided-btn-hint">Teach a fact. e.g., "apples are red"</span>
        </button>
        <button class="guided-btn" data-flow="recognize" title="Recognize a concept from properties. e.g., 'red, round, sweet'">
          Recognize
          <span class="guided-btn-hint">Recognize a concept. e.g., "red, round"</span>
        </button>
        <button class="guided-btn" data-action="perceive" title="Perceive an image and learn from it">
          Perceive
          <span class="guided-btn-hint">Perceive an image</span>
        </button>
        <button class="guided-btn" data-flow="explore" title="Explore the brain's knowledge. e.g., 'apples'">
          Explore
          <span class="guided-btn-hint">Explore the brain's knowledge</span>
        </button>
      </div>
      <div class="guided-btn-row">
        <button class="guided-btn" data-flow="associations" title="List all association types">
          Associations
          <span class="guided-btn-hint">List all association types</span>
        </button>
        <button class="guided-btn" data-flow="define" title="Define a new association type. e.g., 'fruit'">
          Define
          <span class="guided-btn-hint">Define an association. e.g., "fruit"</span>
        </button>
        <button class="guided-btn" data-flow="describe" title="Describe an association with properties. e.g., 'fruit as color, shape'">
          Describe
          <span class="guided-btn-hint">Describe an association. e.g., "fruit as color"</span>
        </button>
      </div>
      <div class="guided-btn-row">
        <button class="guided-btn" data-flow="similar" title="Find neurons similar to one. e.g., 'apple'">
          Similar
          <span class="guided-btn-hint">Find similar neurons. e.g., "apple"</span>
        </button>
        <button class="guided-btn" data-flow="analyze" title="Run a full similarity analysis">
          Analyze
          <span class="guided-btn-hint">Run similarity analysis</span>
        </button>
        <button class="guided-btn" data-flow="stats" title="Show brain statistics">
          Stats
          <span class="guided-btn-hint">Show brain statistics</span>
        </button>
      </div>
      <div class="guided-btn-row">
        <button class="guided-btn" data-flow="neurons" title="List all neurons">
          Neurons
          <span class="guided-btn-hint">List all neurons</span>
        </button>
        <button class="guided-btn" data-flow="paths" title="List all concept paths">
          Paths
          <span class="guided-btn-hint">List all concept paths</span>
        </button>
      </div>
      <div class="guided-btn-row guided-followups" id="guided-followups" style="display:none">
        <button class="guided-btn guided-btn--followup" data-flow="correct" title="Correct a misrecognition. e.g., 'ball'">
          Correct
          <span class="guided-btn-hint">Correct a misrecognition. e.g., "ball"</span>
        </button>
        <button class="guided-btn guided-btn--followup" data-flow="pointout" title="Point out a missed property. e.g., 'seams'">
          Point Out
          <span class="guided-btn-hint">Point out a missed property. e.g., "seams"</span>
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

  // Hide input for flows that don't need it
  const bodyEl = flowEl.querySelector(".guided-flow-body");
  if (def.placeholder === "") {
    bodyEl.style.display = "none";
  } else {
    bodyEl.style.display = "";
  }

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
  flowEl.querySelector(".guided-flow-actions").style.display = "";
  flowEl.querySelector(".guided-flow-progress").style.display = "none";

  // Focus input if visible
  if (def.placeholder !== "") {
    setTimeout(() => input.focus(), 50);
  }

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
  const def = FLOWS[currentFlow];
  const input = flowEl.querySelector(".guided-flow-input");
  const value = input.value.trim();

  // For flows with no placeholder, the input value is not required.
  if (def.placeholder !== "" && !value) return;

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
