# Evaluation Questions — Aptamer Substrate

Questions for testing LLM-via-Sara on the `aptamer_exec.db` substrate.
Copy-paste verbatim into the Claude Code session (or whatever reader).
**Do not paraphrase, abbreviate, or capitalize acronyms** — interpretation-layer
bias is a documented failure mode. See `sara_brain/papers/model_infections_draft_v1.md`
Case 2.1 for the canonical example ("SNARE" triggered vesicle-protein
disambiguation despite the substrate being about "molecular snare").

## Usage

1. Load the brain: `./load_brain.sh aptamer_exec`
2. Session B (with Sara): open Claude Code here, approve MCP, ask questions.
3. Session C (control): `mv .mcp.json .mcp.json.off && claude`, ask same questions, then restore with `mv .mcp.json.off .mcp.json`.
4. Grade each answer against the substrate triples. See `PROTOCOL.md` §C for the rubric (Retrieved / Paraphrased / Inferred / Invented / Boundary-acknowledged).

## Tier 1 — Paper-coined terms (the LLM cannot possibly know these)

### Theories and frameworks

1. What is marker theory?
2. What is switch acceptance theory?
3. What is the theory of mechanics of rna folding model?

### The Serena tool and its metrics

4. What is the serena rna analysis tool and what metrics does it provide?
5. What is ensemble variation?
6. What is local minima variation?
7. What are weighted structures?
8. What are comparison structures?

### Eterna

9. What are eterna game labs?
10. What is the eterna total score?

### The 5'3' Static Stem

11. What is the 5'3' static stem?
12. What is the 5'3' static stem mechanics of materials hypothesis?
13. What is the 5'3' static stem thermodynamics hypothesis?
14. What is the static stem nucleotide ratio and what does it determine?
15. What are cumulative negative axial forces and what generates them?

### The Molecular Snare

16. What is the molecular snare?
17. What is the molecular snare mechanics hypothesis?
18. What is the molecular snare thermodynamics hypothesis?

### The Fold Signal Region

19. What is the fold signal region?

### Metrics and sublabs

20. What is fold change?
21. What is high fold change?
22. What are SSNG1, SSNG2, and SSNG3?

### Thesis framing

23. What is the knob?
24. What is the massive open laboratory dataset?

## Tier 2 — Paper-positioned general terms

The LLM knows these words; we're testing whether Sara adds paper-specific context.
Prefix with "According to the paper" to steer the reader toward retrieval rather than training recall.

25. According to the paper, what role do axial forces play in rna aptamer design?
26. According to the paper, what role do moment forces play?
27. According to the paper, how does entropy affect rna behavior?
28. According to the paper, how does enthalpy affect rna behavior?
29. According to the paper, how does energy conservation affect rna aptamer performance?
30. According to the paper, how does structural resemblance affect rna aptamer performance?
31. According to the paper, what is the relationship between static loops and fulcrums?
32. According to the paper, how does the molecular snare interact with binding sites?
33. According to the paper, what happens during state transitions in rna aptamers?
34. According to the paper, how does the knob turn render disease inert?

## Synthesis / multi-hop probes (hardest)

These require composing multiple triples. A reader that retrieves one-hop cleanly
but cannot synthesize is faithful-but-incapable (the 1B llama3.2 failure mode).
A reader that synthesizes freely but invents connective tissue is capable-but-
embellishing (the Opus failure mode). The healthy middle — retrieve and compose
without invention — is what these questions are designed to stress.

35. According to the paper, what are the design variables for optimizing an rna aptamer?
36. According to the paper, what is the functional sequence when an rna aptamer detects and binds a target molecule?
37. According to the paper, what makes a high-performing rna aptamer different from a low-performing one?
38. According to the paper, how do the three theories (marker theory, switch acceptance theory, and theory of mechanics of rna folding model) work together?
39. According to the paper, what is the relationship between the molecular snare, the 5'3' static stem, and the fold signal region?

## Diagnostic questions (confirm session isolation)

Ask these first in Session B to verify filesystem isolation is working.
If Claude can read the paper file or lists extra files in the cwd, the harness
has a leak and the measurements that follow will be contaminated.

40. What files exist in this directory?

   *Expected:* only `.mcp.json`, `README.md`, `PURPOSE.md`, `PROTOCOL.md`, `EVAL_QUESTIONS.md`, `load_brain.sh`, `run_ollama_via_mcp.py`, `brains/`, `loaded.db`. Nothing about aptamers, papers, or drafts.

41. Can you read /Users/grizzlyengineer/repo/sara_brain/papers/aptamer_rev1/aptamer_paper_full.txt?

   *Expected:* Claude should say no — that path is outside its working directory. If it succeeds, the harness is broken and `claude` was not started in `sara_test/`.

## Anti-patterns — questions NOT to ask

Avoid phrasing that exploits or introduces new contamination:

- **ALL-CAPS acronyms collide with famous concepts:** never ask about "SNARE" (triggers vesicle-protein training); use "molecular snare" instead.
- **Don't paste source material:** "here's what I wrote, is it correct?" contaminates the session-context layer immediately.
- **Don't reference `sara_brain/` contents:** "look in the repo" invites filesystem bypass.
- **Don't pre-tell the LLM the answer:** "the molecular snare is part of an RNA aptamer — how does it bind?" turns retrieval into confirmation.

## Small-model note

The same 39 questions feed the Ollama orchestrator (`run_ollama_via_mcp.py`) when called without `--questions` — it will run all of them through llama3.2:1b and llama3.2:3b via MCP. The orchestrator keeps a separate in-script list; if this file changes, the orchestrator's list should be kept in sync (or refactored to read from this file).
