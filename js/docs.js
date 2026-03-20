/**
 * docs.js — Documentation viewer with changelog.
 *
 * Shows the latest doc for viewing/download, with a version changelog
 * summarizing what changed in each release.
 */

// ── Current Doc ──

const CURRENT_DOC = {
  title: "Sara Brain — Documentation",
  file: "docs/v005_design_philosophy.md",
  filename: "sara_brain_docs_v005.md",
};

// ── Changelog ──

const CHANGELOG = [
  {
    version: "v005",
    date: "2026-03-19",
    label: "current",
    summary: "Design philosophy + user guide: origin story, why paths not activation levels, never forgets, parallel wavefronts, tribal trust, strength formula, 'You Need More Than Attention' (transformers as sensory cortex, inventor skepticism, shared computational roots — 22 academic references) — plus complete REPL reference, data model, perception, LLM translation, all 23 commands",
  },
  {
    version: "v004",
    date: "2026-03-19",
    summary: "Web app guide: guided UI, image viewer with region selection, neural graph visualization, vision proxy setup, data persistence",
  },
  {
    version: "v003",
    date: "2026-03-16",
    summary: "Image perception via Claude Vision, cognitive development model, tribal trust, correction & teaching mechanisms, security & sanitization",
  },
  {
    version: "v002",
    date: "2026-03-14",
    summary: "Full user guide: design philosophy, learning & recognition algorithms, associations & question words, categories, LLM translation, complete REPL reference, storage schema",
  },
  {
    version: "v001",
    date: "2026-03-12",
    summary: "Foundation: architecture overview, data model (neurons, segments, paths), storage schema, initial REPL commands, test suite",
  },
];

// ── State ──

let docContent = null;

// ── DOM Refs ──

let modal = null;
let bodyEl = null;
let changelogEl = null;

// ── Public API ──

export function initDocs() {
  modal = document.getElementById("docs-modal");
  bodyEl = document.getElementById("docs-modal-body");
  changelogEl = document.getElementById("docs-changelog");

  // Build changelog
  changelogEl.innerHTML = "";
  for (const entry of CHANGELOG) {
    const item = document.createElement("div");
    item.className = "changelog-entry" + (entry.label === "current" ? " changelog-current" : "");
    item.innerHTML = `
      <div class="changelog-version">
        ${entry.version}${entry.label === "current" ? ' <span class="changelog-badge">current</span>' : ""}
        <span class="changelog-date">${entry.date}</span>
      </div>
      <div class="changelog-summary">${entry.summary}</div>
    `;
    changelogEl.appendChild(item);
  }

  // Close
  document.getElementById("btn-close-docs").addEventListener("click", hideDocs);

  // Download
  document.getElementById("btn-download-doc").addEventListener("click", () => {
    if (docContent) downloadFile(CURRENT_DOC.filename, docContent);
  });

  // Close on backdrop / escape
  modal.addEventListener("click", (e) => { if (e.target === modal) hideDocs(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.style.display !== "none") hideDocs();
  });
}

export async function showDocs() {
  modal.style.display = "";
  if (!docContent) await loadDoc();
}

export function hideDocs() {
  modal.style.display = "none";
}

// ── Internal ──

async function loadDoc() {
  bodyEl.innerHTML = '<div class="docs-placeholder">Loading...</div>';

  try {
    const resp = await fetch(CURRENT_DOC.file + "?v=" + Date.now());
    if (!resp.ok) throw new Error(`${resp.status}`);
    docContent = await resp.text();
    renderMarkdown(docContent);
  } catch (err) {
    bodyEl.innerHTML = `<div class="docs-placeholder">Error loading document: ${err.message}</div>`;
  }
}

function renderMarkdown(md) {
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

    if (line.trim() === "") { i++; continue; }

    if (/^---+$/.test(line.trim())) { html += "<hr>"; i++; continue; }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      html += `<h${level}>${inlineFormat(headingMatch[2])}</h${level}>`;
      i++;
      continue;
    }

    if (line.trimStart().startsWith("> ")) {
      let quote = "";
      while (i < lines.length && lines[i].trimStart().startsWith("> ")) {
        quote += lines[i].replace(/^>\s?/, "") + " ";
        i++;
      }
      html += `<blockquote>${inlineFormat(quote.trim())}</blockquote>`;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      html += "<ul>";
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        html += `<li>${inlineFormat(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`;
        i++;
      }
      html += "</ul>";
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      html += "<ol>";
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        html += `<li>${inlineFormat(lines[i].replace(/^\s*\d+[.)]\s+/, ""))}</li>`;
        i++;
      }
      html += "</ol>";
      continue;
    }

    html += `<p>${inlineFormat(line)}</p>`;
    i++;
  }

  if (inCodeBlock) {
    html += `<pre class="docs-code"><code>${escapeHtml(codeContent.trimEnd())}</code></pre>`;
  }
  if (inTable) { html += buildTable(tableRows); }

  bodyEl.innerHTML = html;
  bodyEl.scrollTop = 0;
}

function buildTable(rows) {
  if (rows.length === 0) return "";
  let html = '<table class="docs-table"><thead><tr>';
  for (const cell of rows[0]) { html += `<th>${inlineFormat(cell)}</th>`; }
  html += "</tr></thead><tbody>";
  for (let i = 1; i < rows.length; i++) {
    html += "<tr>";
    for (const cell of rows[i]) { html += `<td>${inlineFormat(cell)}</td>`; }
    html += "</tr>";
  }
  html += "</tbody></table>";
  return html;
}

function inlineFormat(text) {
  let s = escapeHtml(text);
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
  s = s.replace(/`([^`]+)`/g, '<code class="docs-inline-code">$1</code>');
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
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
