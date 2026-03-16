/**
 * persistence.js — Save/restore brain state to/from IndexedDB and file export/import.
 */

const DB_NAME = "sara_brain_store";
const DB_VERSION = 1;
const STORE_NAME = "snapshots";
const SNAPSHOT_KEY = "sara_brain_db";

function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Save the current brain state (JSON string) to IndexedDB.
 */
export async function saveBrainState(jsonStr) {
  const db = await openIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    store.put(jsonStr, SNAPSHOT_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Load the saved brain state (JSON string) from IndexedDB.
 * Returns null if no saved state exists.
 */
export async function loadBrainState() {
  const db = await openIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(SNAPSHOT_KEY);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Clear saved brain state from IndexedDB.
 */
export async function clearBrainState() {
  const db = await openIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    store.delete(SNAPSHOT_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Download brain state as a JSON file.
 */
export function downloadBrainExport(jsonStr) {
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sara_brain_export.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Trigger a file upload dialog and return the file contents as text.
 */
export function uploadBrainImport() {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = () => {
      const file = input.files[0];
      if (!file) {
        reject(new Error("No file selected"));
        return;
      }
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsText(file);
    };
    input.click();
  });
}
