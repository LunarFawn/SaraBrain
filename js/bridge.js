/**
 * bridge.js — JS↔Python bridge via Pyodide.
 * All calls go through pyodide.runPythonAsync() and convert results to JS.
 */

let pyodide = null;

/**
 * Set the Pyodide instance (called from app.js after loading).
 */
export function setPyodide(py) {
  pyodide = py;
}

/**
 * Initialize the brain (call boot.init_brain()).
 */
export async function initBrain() {
  return await pyodide.runPythonAsync("init_brain()");
}

/**
 * Run a REPL command and return the output string.
 */
export async function runCommand(commandLine) {
  // Escape backslashes and quotes for safe Python string embedding
  const escaped = commandLine.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
  const result = await pyodide.runPythonAsync(`run_command('${escaped}')`);
  return result;
}

/**
 * Get graph data (nodes + links) as a JS object.
 */
export async function getGraphData() {
  const jsonStr = await pyodide.runPythonAsync("get_graph_data()");
  return JSON.parse(jsonStr);
}

/**
 * Get cached recognition path data for wavefront animation.
 * Must be called after runCommand('recognize ...') — does NOT re-run recognition.
 */
export async function getLastRecognitionPaths() {
  const jsonStr = await pyodide.runPythonAsync("get_last_recognition_paths()");
  return JSON.parse(jsonStr);
}

/**
 * Export brain state as a JSON string.
 */
export async function exportBrain() {
  return await pyodide.runPythonAsync("export_db()");
}

/**
 * Import brain state from a JSON string.
 */
export async function importBrain(jsonStr) {
  // Use a global variable to avoid string escaping issues
  pyodide.globals.set("_import_data", jsonStr);
  const result = await pyodide.runPythonAsync("import_db(_import_data)");
  pyodide.globals.delete("_import_data");
  return result;
}

/**
 * Seed the brain with demo data.
 */
export async function seedBrain() {
  return await pyodide.runPythonAsync("_seed_brain()");
}

/**
 * Seed the brain with Wikipedia demo data.
 * Fetches the JSON from the server and passes it to Python.
 */
export async function seedWiki() {
  const resp = await fetch(`python/wiki_demo_brain.json?v=${Date.now()}`);
  if (!resp.ok) throw new Error(`Failed to fetch wiki demo: ${resp.status}`);
  const jsonStr = await resp.text();
  pyodide.globals.set("_wiki_data", jsonStr);
  const result = await pyodide.runPythonAsync("_seed_wiki(_wiki_data)");
  pyodide.globals.delete("_wiki_data");
  return result;
}

/**
 * Get question words (association → question word mappings) as a JS object.
 */
export async function getQuestionWords() {
  const json = await pyodide.runPythonAsync("get_question_words()");
  return JSON.parse(json);
}

/**
 * Get known properties for a candidate concept label.
 */
export async function getCandidateProperties(label) {
  const escaped = label.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
  const json = await pyodide.runPythonAsync(`get_candidate_properties('${escaped}')`);
  return JSON.parse(json);
}

/**
 * Store perception state from JS into Python for no/see commands.
 */
export async function setPerceptionState(stateObj) {
  const json = JSON.stringify(stateObj);
  pyodide.globals.set("_ps_json", json);
  await pyodide.runPythonAsync("set_perception_state(_ps_json)");
  pyodide.globals.delete("_ps_json");
}
