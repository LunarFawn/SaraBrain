/**
 * docs.js — Documentation viewer and download manager.
 *
 * Fetches markdown docs, renders them with a lightweight parser,
 * and provides download functionality.
 */

// ── Doc Registry ──

const DOCS = [
  {
    id: "v001",
    title: "v001 — Foundation",
    file: "docs/v001_foundation.md",
    description: "Architecture, data model, storage schema",
  },
  {
    id: "v002",
    title: "v002 — User Guide",
    file: "docs/v002_user_guide.md",
    description: "Philosophy, algorithms, associations, categories, LLM, REPL reference",
  },
  {
    id: "v003",
    title: "v003 — Perception",
    file: "docs/v003_perception.md",
    description: "Vision perception, cognitive development, tribal trust, security",
  },
  {
    id: "v004",
    title: "v004 — Web App",
    file: "docs/v004_web_app.md",
    description: "Guided UI, image viewer, region selection, neural graph, vision proxy",
  },
];

// ── State ──

const docCache = {};
let currentDoc = null;

// ── DOM Refs ──

let modal = null;
let docsList = null;
let bodyEl = null;
let titleEl = null;
let downloadBtn = null;

// ── Public API ──

export function initDocs() {
  modal = document.getElementById("docs-modal");
  docsList = document.getElementById("docs-list");
  bodyEl = document.getElementById("docs-modal-body");
  titleEl = document.getElementById("docs-current-title");
  downloadBtn = document.getElementById("btn-download-doc");

  // Build sidebar list
  docsList.innerHTML = "";
  for (const doc of DOCS) {
    const item = document.createElement("div");
    item.className = "docs-list-item";
    item.dataset.id = doc.id;
    item.innerHTML = `
      <div class="docs-list-item-title">${doc.title}</div>
      <div class="docs-list-item-desc">${doc.description}</div>
    `;
    item.addEventListener("click", () => loadDoc(doc));
    docsList.appendChild(item);
  }

  // Close button
  document.getElementById("btn-close-docs").addEventListener("click", hideDocs);

  // Download current doc
  downloadBtn.addEventListener("click", () => {
    if (currentDoc && docCache[currentDoc.id]) {
      downloadFile(currentDoc.file.split("/").pop(), docCache[currentDoc.id]);
    }
  });

  // Download all as zip (simple: downloads each file individually)
  document.getElementById("btn-download-all-docs").addEventListener("click", downloadAllDocs);

  // Close on backdrop click
  modal.addEventListener("click", (e) => {
    if (e.target === modal) hideDocs();
  });

  // Close on Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.style.display !== "none") hideDocs();
  });
}

export function showDocs() {
  modal.style.display = "";
}

export function hideDocs() {
  modal.style.display = "none";
}

// ── Internal ──

async function loadDoc(doc) {
  // Update sidebar selection
  for (const item of docsList.querySelectorAll(".docs-list-item")) {
    item.classList.toggle("active", item.dataset.id === doc.id);
  }

  currentDoc = doc;
  titleEl.textContent = doc.title;
  downloadBtn.style.display = "";

  // Check cache
  if (docCache[doc.id]) {
    renderMarkdown(docCache[doc.id]);
    return;
  }

  // Show loading
  bodyEl.innerHTML = '<div class="docs-placeholder">Loading...</div>';

  try {
    const resp = await fetch(doc.file);
    if (!resp.ok) throw new Error(`Failed to fetch ${doc.file}: ${resp.status}`);
    const text = await resp.text();
    docCache[doc.id] = text;
    renderMarkdown(text);
  } catch (err) {
    bodyEl.innerHTML = `<div class="docs-placeholder">Error loading document: ${err.message}</div>`;
  }
}

function renderMarkdown(md) {
  // Lightweight markdown → HTML conversion
  // Handles: headings, code blocks, inline code, tables, bold, italic, links, lists, blockquotes, hr
  let html = "";
  const lines = md.split("\n");
  let i = 0;
  let inCodeBlock = false;
  let codeContent = "";
  let inTable = false;
  let tableRows = [];

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code blocks
    if (line.trimStart().startsWith("```")) {
      if (inCodeBlock) {
        html += `<pre class="docs-code"><code>${escapeHtml(codeContent.trimEnd())}</code></pre>`;
        codeContent = "";
        inCodeBlock = false;
      } else {
        // Flush any open table
        if (inTable) { html += buildTable(tableRows); inTable = false; tableRows = []; }
        inCodeBlock = true;
      }
      i++;
      continue;
    }

    if (inCodeBlock) {
      codeContent += line + "\n";
      i++;
      continue;
    }

    // Table rows
    if (line.includes("|") && line.trim().startsWith("|")) {
      if (!inTable) { inTable = true; tableRows = []; }
      // Skip separator rows (|---|---|)
      if (/^\|[\s-:|]+\|$/.test(line.trim())) { i++; continue; }
      const cells = line.split("|").slice(1, -1).map(c => c.trim());
      tableRows.push(cells);
      i++;
      continue;
    } else if (inTable) {
      html += buildTable(tableRows);
      inTable = false;
      tableRows = [];
    }

    // Blank line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Horizontal rule
    if (/^---+$/.test(line.trim())) {
      html += "<hr>";
      i++;
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,6})\s+(.+)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      html += `<h${level}>${inlineFormat(headingMatch[2])}</h${level}>`;
      i++;
      continue;
    }

    // Blockquote
    if (line.trimStart().startsWith("> ")) {
      let quote = "";
      while (i < lines.length && lines[i].trimStart().startsWith("> ")) {
        quote += lines[i].replace(/^>\s?/, "") + " ";
        i++;
      }
      html += `<blockquote>${inlineFormat(quote.trim())}</blockquote>`;
      continue;
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      html += "<ul>";
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        html += `<li>${inlineFormat(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`;
        i++;
      }
      html += "</ul>";
      continue;
    }

    // Ordered list
    if (/^\s*\d+[.)]\s+/.test(line)) {
      html += "<ol>";
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        html += `<li>${inlineFormat(lines[i].replace(/^\s*\d+[.)]\s+/, ""))}</li>`;
        i++;
      }
      html += "</ol>";
      continue;
    }

    // Paragraph
    html += `<p>${inlineFormat(line)}</p>`;
    i++;
  }

  // Flush
  if (inCodeBlock) {
    html += `<pre class="docs-code"><code>${escapeHtml(codeContent.trimEnd())}</code></pre>`;
  }
  if (inTable) {
    html += buildTable(tableRows);
  }

  bodyEl.innerHTML = html;
  bodyEl.scrollTop = 0;
}

function buildTable(rows) {
  if (rows.length === 0) return "";
  let html = '<table class="docs-table"><thead><tr>';
  for (const cell of rows[0]) {
    html += `<th>${inlineFormat(cell)}</th>`;
  }
  html += "</tr></thead><tbody>";
  for (let i = 1; i < rows.length; i++) {
    html += "<tr>";
    for (const cell of rows[i]) {
      html += `<td>${inlineFormat(cell)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  return html;
}

function inlineFormat(text) {
  let s = escapeHtml(text);
  // Bold
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic
  s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
  // Inline code
  s = s.replace(/`([^`]+)`/g, '<code class="docs-inline-code">$1</code>');
  // Links [text](url)
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function downloadFile(filename, content) {
  const blob = new Blob([content], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function downloadAllDocs() {
  // Fetch all docs and download each one
  for (const doc of DOCS) {
    let content = docCache[doc.id];
    if (!content) {
      try {
        const resp = await fetch(doc.file);
        content = await resp.text();
        docCache[doc.id] = content;
      } catch {
        continue;
      }
    }
    downloadFile(doc.file.split("/").pop(), content);
    // Small delay between downloads so browser doesn't block them
    await new Promise(r => setTimeout(r, 300));
  }
}
