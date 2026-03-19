/**
 * vision.js — JS vision client + sanitization + perception loop.
 *
 * Port of Python VisionObserver + Perceiver to JavaScript.
 * All Claude output is sanitized to safe property labels before reaching the brain.
 */

// ── Label pattern (exact match of Python _LABEL_PATTERN) ──

const LABEL_PATTERN = /^[a-z0-9][a-z0-9_ -]*$/;
const BLOCKED_KEYWORDS = ["http", "www", "import ", "def ", "class ", "print("];
const MAX_LABEL_LENGTH = 40;

/**
 * Sanitize raw Claude output to safe lowercase property labels.
 * Exact port of Python VisionObserver._sanitize().
 */
export function sanitize(rawText) {
  const labels = [];
  for (const line of rawText.split("\n")) {
    // Strip bullets, dashes, numbers, colons
    let cleaned = line.trim().replace(/^[-*•·0-9.)]+/, "");
    // Remove common prefixes like "color:"
    if (cleaned.includes(":")) {
      cleaned = cleaned.split(":").pop();
    }
    cleaned = cleaned.trim().toLowerCase();
    // Split on commas for multi-value lines
    for (let part of cleaned.split(",")) {
      part = part.trim();
      // Remove surrounding quotes
      part = part.replace(/^["'`]+|["'`]+$/g, "");
      // Only allow simple label characters
      part = part.replace(/[^a-z0-9_ -]/g, "");
      part = part.trim();
      if (part && LABEL_PATTERN.test(part) && part.length <= MAX_LABEL_LENGTH) {
        if (!BLOCKED_KEYWORDS.some((kw) => part.includes(kw))) {
          labels.push(part);
        }
      }
    }
  }
  // Deduplicate preserving order
  const seen = new Set();
  const result = [];
  for (const label of labels) {
    if (!seen.has(label)) {
      seen.add(label);
      result.push(label);
    }
  }
  return result;
}

// ── API Calls ──

/**
 * Call the Claude Vision API via the local proxy.
 * Returns the text response or null on error.
 */
export async function callVision(proxyUrl, apiKey, model, imageBase64, mediaType, prompt, maxTokens = 300) {
  const payload = {
    model,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: mediaType, data: imageBase64 },
          },
          { type: "text", text: prompt },
        ],
      },
    ],
    temperature: 0,
    max_tokens: maxTokens,
  };

  try {
    const resp = await fetch(`${proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      console.warn("Vision API error:", resp.status, await resp.text());
      return null;
    }
    const body = await resp.json();
    return body.content[0].text.trim();
  } catch (err) {
    console.warn("Vision API call failed:", err);
    return null;
  }
}

/**
 * Phase 1: Freely describe everything visible in the image.
 */
export async function observeInitial(proxyUrl, apiKey, model, imageBase64, mediaType) {
  const prompt =
    "Describe everything you observe in this image. " +
    "List each observation as a single word or short phrase on its own line. " +
    "Include: colors, shapes, textures, patterns, materials, objects, " +
    "features, markings, and any distinguishing characteristics. " +
    "Be thorough — report everything you see, even subtle details. " +
    "One observation per line, lowercase, simple words only.";
  const raw = await callVision(proxyUrl, apiKey, model, imageBase64, mediaType, prompt);
  if (raw === null) return [];
  return sanitize(raw);
}

/**
 * Phase 2: Ask targeted questions about specific associations.
 * questions: {association: questionText}
 * Returns: {association: value|null}
 */
export async function observeDirected(proxyUrl, apiKey, model, imageBase64, mediaType, questions) {
  if (Object.keys(questions).length === 0) return {};

  const lines = Object.entries(questions).map(([assoc, q]) => `${assoc}: ${q}`);
  const prompt =
    "Answer each question about this image. " +
    "For each, give a single-word or short-phrase answer. " +
    "If you cannot determine the answer from the image, say 'cannot determine'.\n\n" +
    lines.join("\n");

  const raw = await callVision(proxyUrl, apiKey, model, imageBase64, mediaType, prompt);
  if (raw === null) {
    const empty = {};
    for (const assoc of Object.keys(questions)) empty[assoc] = null;
    return empty;
  }

  const results = {};
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.includes(":")) continue;
    const colonIdx = trimmed.indexOf(":");
    const key = trimmed.slice(0, colonIdx).trim().toLowerCase();
    const value = trimmed.slice(colonIdx + 1).trim().toLowerCase();

    for (const assoc of Object.keys(questions)) {
      if (key.includes(assoc)) {
        if (value.includes("cannot determine") || value.includes("not possible") || value.includes("unknown")) {
          results[assoc] = null;
        } else {
          const sanitized = sanitize(value);
          results[assoc] = sanitized.length > 0 ? sanitized[0] : null;
        }
        break;
      }
    }
  }

  // Fill missing
  for (const assoc of Object.keys(questions)) {
    if (!(assoc in results)) results[assoc] = null;
  }
  return results;
}

/**
 * Phase 3: Verify a single property — YES/NO/indeterminate.
 */
export async function verifyProperty(proxyUrl, apiKey, model, imageBase64, mediaType, prop) {
  const prompt =
    `Does this image appear to show something that is '${prop}'? ` +
    "Answer only YES, NO, or CANNOT DETERMINE.";
  const raw = await callVision(proxyUrl, apiKey, model, imageBase64, mediaType, prompt, 20);
  if (raw === null) return null;
  const answer = raw.trim().toUpperCase();
  if (answer.includes("CANNOT") || answer.includes("DETERMINE")) return null;
  if (answer.includes("YES")) return true;
  if (answer.includes("NO")) return false;
  return null;
}

/**
 * Check if the local proxy is running.
 */
export async function checkProxyHealth(proxyUrl) {
  try {
    const resp = await fetch(`${proxyUrl}/health`, { signal: AbortSignal.timeout(2000) });
    if (!resp.ok) return false;
    const body = await resp.json();
    return body.status === "ok";
  } catch {
    return false;
  }
}

// ── SHA-256 for label generation ──

async function sha256hex(arrayBuffer) {
  const hash = await crypto.subtle.digest("SHA-256", arrayBuffer);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Generate a label from filename + image hash.
 */
export async function generateLabel(fileName, arrayBuffer) {
  const stem = fileName.replace(/\.[^.]+$/, "").toLowerCase().replace(/\s+/g, "_");
  const hash = await sha256hex(arrayBuffer);
  return `img_${stem}_${hash.slice(0, 6)}`;
}

// ── Perception Loop ──

/**
 * Run the full perception loop (mirrors Python Perceiver.perceive()).
 *
 * bridge: { runCommand, getQuestionWords, getCandidateProperties, setPerceptionState }
 * onStep: callback({phase, observations, recognition, taughtCount}) for live display
 *
 * Returns: {label, steps, allObservations, totalTaught, finalRecognition}
 */
export async function runPerceptionLoop(proxyUrl, apiKey, model, imageBase64, mediaType, label, bridge, onStep) {
  const allObserved = [];
  let prevTop = null;
  let prevConfidence = 0;
  let totalTaught = 0;
  const steps = [];

  // Create the concept neuron
  await bridge.runCommand(`teach ${label} is ${label}`);

  // --- Phase 1: Initial Observation ---
  const observations = await observeInitial(proxyUrl, apiKey, model, imageBase64, mediaType);

  let taught = 0;
  for (const prop of observations) {
    await bridge.runCommand(`teach ${label} is ${prop}`);
    taught++;
  }
  allObserved.push(...observations);
  totalTaught += taught;

  let recognition = "";
  if (allObserved.length > 0) {
    recognition = await bridge.runCommand(`recognize ${allObserved.join(", ")}`);
  }

  const step1 = { phase: "initial", observations, recognition, taughtCount: taught };
  steps.push(step1);
  if (onStep) onStep(step1);

  // Track convergence from recognition output
  const topMatch1 = parseTopRecognition(recognition);
  if (topMatch1) {
    prevTop = topMatch1.label;
    prevConfidence = topMatch1.confidence;
  }

  // --- Phase 2: Directed Inquiry (up to 3 rounds) ---
  for (let round = 0; round < 3; round++) {
    let qwords;
    try {
      qwords = await bridge.getQuestionWords();
    } catch {
      break;
    }
    if (!qwords || Object.keys(qwords).length === 0) break;

    // Gather all associations
    const allAssocs = [...new Set(Object.values(qwords).flat())].sort();
    // Filter to unobserved (simple heuristic: skip if any observed prop matches assoc name)
    const unobserved = allAssocs.filter((a) => !allObserved.includes(a));
    if (unobserved.length === 0) break;

    // Build questions
    const questions = {};
    for (const assoc of unobserved) {
      questions[assoc] = `What ${assoc} does this appear to have or be?`;
    }

    const directedResults = await observeDirected(proxyUrl, apiKey, model, imageBase64, mediaType, questions);

    const newObs = [];
    taught = 0;
    for (const [assoc, value] of Object.entries(directedResults)) {
      if (value !== null) {
        newObs.push(value);
        await bridge.runCommand(`teach ${label} is ${value}`);
        taught++;
      }
    }
    allObserved.push(...newObs);
    totalTaught += taught;

    recognition = "";
    if (allObserved.length > 0) {
      recognition = await bridge.runCommand(`recognize ${allObserved.join(", ")}`);
    }

    const stepN = { phase: `directed-${round + 1}`, observations: newObs, recognition, taughtCount: taught };
    steps.push(stepN);
    if (onStep) onStep(stepN);

    // Check convergence
    const topMatchN = parseTopRecognition(recognition);
    if (topMatchN) {
      if (topMatchN.label === prevTop && topMatchN.confidence === prevConfidence) break;
      prevTop = topMatchN.label;
      prevConfidence = topMatchN.confidence;
    }
  }

  // --- Phase 3: Verification ---
  if (prevTop && prevTop !== label) {
    let candidateProps;
    try {
      candidateProps = await bridge.getCandidateProperties(prevTop);
    } catch {
      candidateProps = [];
    }

    const verifiedObs = [];
    taught = 0;
    for (const prop of candidateProps) {
      if (allObserved.includes(prop)) continue;
      const verified = await verifyProperty(proxyUrl, apiKey, model, imageBase64, mediaType, prop);
      if (verified === true) {
        verifiedObs.push(prop);
        await bridge.runCommand(`teach ${label} is ${prop}`);
        taught++;
      }
    }
    allObserved.push(...verifiedObs);
    totalTaught += taught;

    if (verifiedObs.length > 0 && allObserved.length > 0) {
      recognition = await bridge.runCommand(`recognize ${allObserved.join(", ")}`);
    }

    const stepV = { phase: "verification", observations: verifiedObs, recognition, taughtCount: taught };
    steps.push(stepV);
    if (onStep) onStep(stepV);
  }

  const result = {
    label,
    steps,
    allObservations: allObserved,
    totalTaught,
    finalRecognition: recognition,
  };

  // Store perception state for no/see commands
  await bridge.setPerceptionState({
    label,
    all_observations: allObserved,
    top_guess: prevTop,
  });

  return result;
}

/**
 * Parse the top recognition result from text output.
 * Looks for pattern like "  concept (confidence: N, ...)"
 */
function parseTopRecognition(text) {
  if (!text) return null;
  const match = text.match(/^\s+(\S+)\s+\(confidence:\s*(\d+)/m);
  if (match) {
    return { label: match[1], confidence: parseInt(match[2], 10) };
  }
  return null;
}
