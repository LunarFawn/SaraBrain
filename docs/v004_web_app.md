# Sara Brain v004 — Web App & Interactive UI

> Sara Brain runs entirely in your browser — no server, no install, no data leaves your machine (except Vision API calls you explicitly authorize).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Guided Mode](#guided-mode)
5. [REPL Mode](#repl-mode)
6. [Neural Graph Visualization](#neural-graph-visualization)
7. [Image Viewer & Region Selection](#image-viewer--region-selection)
8. [Vision Setup & Perception](#vision-setup--perception)
9. [Vision Proxy](#vision-proxy)
10. [Data Persistence & Portability](#data-persistence--portability)
11. [Complete Feature Reference](#complete-feature-reference)

---

## Overview

The Sara Brain web app is a fully interactive brain simulation that runs in your browser. It uses **Pyodide** to run the real Python sara_brain engine inside WebAssembly — the same code that runs on your machine. The neural graph is rendered with **D3.js**, and the entire state persists to your browser's localStorage.

### What you can do

- **Teach** Sara facts and watch neurons appear in the graph
- **Recognize** concepts from properties with animated wavefront paths
- **Perceive** images through Claude Vision with live-streaming label chips
- **Select regions** on images to ask Claude about specific areas
- **Explore** the brain with why/trace queries
- **Correct** Sara's mistakes and point out what she missed
- **Export/import** the full brain state as JSON

### What stays local

Everything except Vision API calls runs entirely in your browser. No server processes your data. The optional Vision feature sends image data to the Anthropic API through a small local proxy you run yourself — your API key, your choice.

---

## Architecture

### How Python runs in the browser

```
Browser
├── index.html            ← Page structure
├── js/app.js             ← Main orchestrator
├── js/bridge.js          ← JavaScript ↔ Python bridge via Pyodide
├── js/graph.js           ← D3.js force-directed neural graph
├── js/guided.js          ← Guided mode UI (action buttons, flows)
├── js/imageviewer.js     ← Canvas-based image viewer with selection tools
├── js/vision.js          ← Vision API client (JS port of Python VisionObserver)
├── js/persistence.js     ← localStorage save/restore
└── python/
    ├── boot.py           ← Pyodide initialization script
    └── sara_brain/       ← Full Python package (fetched at boot)
```

### Boot sequence

1. **Load Pyodide** — WebAssembly Python runtime (~10 MB, cached by browser)
2. **Load sqlite3** — Pyodide package for in-memory SQLite
3. **Fetch sara_brain** — All Python source files fetched from `python/` and written to Pyodide's virtual filesystem
4. **Run boot.py** — Initializes the Python environment, creates `Brain` instance
5. **Restore state** — Loads saved JSON from localStorage and imports it, or seeds demo data on first visit
6. **Init UI** — Wires up all event listeners, initializes graph, guided mode, and image viewer

### The bridge

`js/bridge.js` is the translation layer between JavaScript and Python. Every REPL command goes through this bridge:

```
User clicks "Teach" → guided.js calls onTeach("dogs have fur")
  → app.js calls executeCommand("teach dogs have fur")
    → bridge.js calls pyodide.runPythonAsync("run_command('teach dogs have fur')")
      → Python sara_brain processes the command
    → bridge.js returns the text output
  → app.js displays output in the activity log
  → app.js calls refreshGraph() → bridge.js fetches graph data → D3 renders
```

---

## Getting Started

### Online (GitHub Pages)

Visit the deployed app at your GitHub Pages URL. The app loads automatically — Pyodide and sara_brain are fetched on first visit (subsequent visits use browser cache).

### Local development

```bash
# Serve from the gh-pages branch
git checkout gh-pages
python3 -m http.server 8000

# Open http://localhost:8000
```

### First visit

On your first visit, the app automatically seeds demo data (fruits, shapes, colors) so you have something to explore immediately. The seeded data appears in the neural graph and you can start teaching, recognizing, and exploring right away.

---

## Guided Mode

Guided mode is the default interface. It replaces the command-line REPL with action buttons that walk you through each operation.

### Action buttons

| Button | What it does | What you type |
|--------|-------------|---------------|
| **Teach** | Opens a flow to teach Sara a fact | `apples are red` |
| **Recognize** | Opens a flow to recognize from properties | `red, round, sweet` |
| **Perceive** | Opens a file picker to show Sara an image | *(selects image file)* |
| **Explore** | Opens a flow with Why and Trace sub-buttons | `apples` |

### How flows work

1. Click an action button (e.g., **Teach**)
2. An inline panel appears with a text input, hint, and submit/cancel buttons
3. Type your input (e.g., `dogs have fur`)
4. Click the submit button or press Enter
5. The panel shows "Working..." while the command executes
6. Results appear in the activity log below
7. The flow auto-closes on completion

### Contextual follow-up buttons

After perceiving an image, two additional buttons appear:

| Button | What it does | When to use |
|--------|-------------|-------------|
| **Correct** | Tell Sara it guessed wrong | Sara said "apple" but it's a ball |
| **Point Out** | Teach Sara a property it missed | Sara didn't notice the seams |

These buttons disappear when you reset the brain.

### Hover hints

Every action button has a hover hint that appears below it describing what the button does. Native browser tooltips also appear on hover.

---

## REPL Mode

Click the **REPL** toggle at the top of the left panel to switch to REPL mode. This gives you the same text-input command line as the Python CLI.

### Available commands

All commands from the Python REPL work in the web app:

| Command | Example |
|---------|---------|
| `teach <statement>` | `teach a dog is furry` |
| `recognize <properties>` | `recognize red, round` |
| `trace <label>` | `trace red` |
| `why <label>` | `why apple` |
| `similar <label>` | `similar red` |
| `analyze` | `analyze` |
| `define <name> <qword>` | `define mood how` |
| `describe <name> as <props>` | `describe mood as happy, sad` |
| `<qword> <concept> <assoc>` | `what apple color` |
| `categorize <concept> <cat>` | `categorize apple item` |
| `perceive` | *(opens file picker)* |
| `no <correct_label>` | `no ball` |
| `see <property>` | `see seams` |
| `neurons` | `neurons` |
| `paths` | `paths` |
| `stats` | `stats` |
| `seed` | `seed` |
| `reset` | `reset` |

### Keyboard shortcuts

- **Enter** — Execute command
- **Up arrow** — Previous command in history
- **Down arrow** — Next command in history

### Shared output log

Both Guided mode and REPL mode share the same activity log. Switching modes does not clear the log — you can teach something in Guided mode and see the result when you switch to REPL mode.

---

## Neural Graph Visualization

The right panel displays a D3.js force-directed graph of all neurons and segments in the brain.

### Node types and colors

| Type | Color | Shape |
|------|-------|-------|
| Concept | Light blue | Circle |
| Property | Pink | Circle |
| Relation | Yellow | Circle |
| Association | Purple | Circle |

### Interactions

- **Drag** nodes to rearrange the graph layout
- **Zoom** with scroll wheel or pinch
- **Pan** by dragging the background
- **Click** a node to run `why` (for concepts) or `trace` (for properties) — results appear in the detail panel at the bottom

### Wavefront animation

When you run `recognize`, the graph animates the parallel wavefronts. Each path lights up sequentially, showing how the wavefronts propagate from input properties through relation neurons to concept intersections.

### Graph/Image tabs

After perceiving an image, a second tab appears in the graph header:

- **Graph** — Shows the neural graph (default)
- **Image** — Shows the uploaded image with label chips and selection tools

---

## Image Viewer & Region Selection

When you perceive an image, it appears in the right panel with an interactive canvas.

### Label chips

As Sara perceives the image, color-coded label chips stream in above the canvas:

| Phase | Color | Meaning |
|-------|-------|---------|
| Initial observation | Green | Claude freely described these |
| Directed inquiry | Blue | Sara asked Claude about specific associations |
| Verification | Yellow | Sara verified properties of its top guess |
| Region query | Purple | You selected a region and Claude described it |

### Selection tools

Below the canvas, three tools let you select regions to query Claude about:

| Tool | How to use | What happens |
|------|-----------|--------------|
| **Point** | Click on the image | 200x200 pixel crop centered on your click |
| **Rect** | Click and drag a rectangle | Crops the selected rectangle |
| **Lasso** | Click and draw a freeform shape | Crops the bounding box of your shape |

### Region queries

After selecting a region:
1. The coordinates appear in the toolbar (e.g., `(120, 80) 200x150`)
2. The cropped region is sent to Claude Vision
3. Claude describes what it sees in the region
4. Each observation is taught as a property of the image
5. Purple label chips appear for the new observations
6. The neural graph updates with the new facts

### How it works technically

The image viewer uses a two-canvas stack:
- **Lower canvas** — Displays the image, letterboxed to fit the container while preserving aspect ratio
- **Upper canvas** — Captures pointer events and draws selection visuals (translucent blue rectangles or lasso paths)

All crops happen at the image's original native resolution using an offscreen canvas, regardless of how the image is displayed on screen. This ensures Claude gets the highest quality crop for analysis.

---

## Vision Setup & Perception

Vision requires three things: the local proxy running, an Anthropic API key, and a model selection.

### Step-by-step setup

1. Click **Vision** in the header to expand the Vision panel
2. Download or copy the proxy script (see [Vision Proxy](#vision-proxy) below)
3. Run the proxy: `python vision_proxy.py`
4. The status dot turns green when the proxy connects
5. Enter your Anthropic API key (stored in localStorage, never sent anywhere except through your local proxy)
6. Optionally change the model (default: `claude-sonnet-4-20250514`)

### Running perception

**In Guided mode:** Click the **Perceive** button → select an image file.

**In REPL mode:** Type `perceive` → select an image file.

### What happens during perception

1. The image loads into the Image Viewer and the right panel switches to the Image tab
2. **Phase 1 — Initial observation**: Claude freely describes everything visible. Green label chips stream in. Sara teaches each observation as a fact. Recognition runs.
3. **Phase 2 — Directed inquiry** (up to 3 rounds): Sara asks Claude about known associations (color, texture, etc.) it hasn't observed yet. Blue chips stream in.
4. **Phase 3 — Verification**: Sara looks up properties of its top guess and asks Claude to verify them. Yellow chips stream in.
5. The **Correct** and **Point Out** buttons appear for follow-up.
6. The neural graph updates with all the new neurons and paths.

### Correcting Sara

After perception, if Sara guessed wrong:

1. Click **Correct** (or type `no <correct_label>` in REPL)
2. Enter the correct identity (e.g., `ball`)
3. Sara retains all original observations but now teaches them under the correct concept
4. Both the wrong guess and correct answer share the same observed properties — next time Sara must find more distinguishing properties

### Pointing out missed details

If Sara missed something:

1. Click **Point Out** (or type `see <property>` in REPL)
2. Enter the missed property (e.g., `seams`)
3. Sara learns this property for the current image

---

## Vision Proxy

The vision proxy is a small Python script (~60 lines, no dependencies) that runs on your machine and forwards requests from the browser to the Anthropic API, adding CORS headers so the browser allows the cross-origin request.

### Why a proxy?

Browsers block direct requests to `api.anthropic.com` due to CORS (Cross-Origin Resource Sharing) restrictions. The proxy runs locally on `http://localhost:8765` and forwards your requests, adding the necessary headers.

### Setup

```bash
# Download from the web app (click "Download Proxy" in the Vision panel)
# Or copy from the repo:
cp src/sara_brain/vision_proxy.py .

# Run it:
python vision_proxy.py

# You should see:
# Vision proxy running on http://localhost:8765
# Press Ctrl+C to stop
```

### What it does

- Listens on `http://localhost:8765`
- `/health` — Returns `{"status": "ok"}` (the web app polls this to check connectivity)
- `OPTIONS *` — Returns CORS preflight headers (required by browsers)
- `POST /v1/messages` — Forwards to `https://api.anthropic.com/v1/messages` with your API key, adds CORS headers to the response

### Security

- The proxy runs on **localhost only** — not accessible from other machines
- Your API key travels from browser → localhost proxy → Anthropic API. It never touches any third-party server
- The proxy source code is viewable directly in the web app (click "View Code" in the Vision panel)
- No dependencies — pure Python stdlib (`http.server`, `urllib.request`, `json`)

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Red dot (not connected) | Make sure `python vision_proxy.py` is running |
| Port already in use | Another process is on port 8765. Kill it or edit the port in the script |
| "Perceive Image" button disabled | Both proxy connection AND API key are required |
| CORS errors in console | The proxy isn't running, or you're hitting the API directly |
| API errors (401, 429) | Check your API key, check your Anthropic usage limits |

---

## Data Persistence & Portability

### Automatic save

Every time you teach, recognize, perceive, or modify the brain, the full state is automatically saved to your browser's **localStorage**. No manual save needed.

### Export

Click **Export** in the header to download a JSON file containing the complete brain state — all neurons, segments, paths, associations, categories, and settings. The file can be loaded on any machine.

### Import

Click **Import** to load a previously exported JSON file. This replaces the current brain state.

### Reset

Click **Reset** to clear the entire brain and start fresh. This also clears localStorage. On next load, demo data will be seeded again.

### Seed Demo

Click **Seed Demo** to populate the brain with example data (fruits, shapes, colors) without resetting existing data.

### Storage limits

localStorage typically allows 5-10 MB per origin. A brain with hundreds of neurons and thousands of paths will fit comfortably. If you approach the limit, export regularly.

---

## Complete Feature Reference

### Header buttons

| Button | Action |
|--------|--------|
| **Seed Demo** | Populate brain with example data |
| **Reset** | Clear brain, localStorage, and image viewer |
| **Export** | Download brain state as JSON |
| **Import** | Upload and restore a JSON brain file |
| **Vision** | Toggle the Vision setup panel |

### Left panel

| Element | Description |
|---------|-------------|
| **Guided/REPL toggle** | Switch between button-driven and command-line interfaces |
| **Action buttons** | Teach, Recognize, Perceive, Explore (Guided mode) |
| **Follow-up buttons** | Correct, Point Out (appear after perception) |
| **Activity log** | Scrollable output from all commands (shared by both modes) |
| **Command input** | Text input with history (REPL mode) |

### Right panel

| Element | Description |
|---------|-------------|
| **Graph/Image tabs** | Toggle between neural graph and image viewer |
| **Neural graph** | D3.js force-directed visualization of all neurons |
| **Image viewer** | Canvas display with label chips and selection tools |
| **Selection toolbar** | Point, Rect, Lasso tools with coordinate display |

### Bottom panel

| Element | Description |
|---------|-------------|
| **Detail panel** | Shows trace/why results and node click details |

### Test counts

166 tests across 16 test files covering the full Python engine, vision system, and proxy.
