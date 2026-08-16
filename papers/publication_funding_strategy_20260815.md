# Publication & Funding Strategy — Sara Brain
**Date: 2026-08-15**

## Publication Targets

### Conferences
- **NeurIPS** — neuro-symbolic AI track (deadline ~May/June)
- **AAAI** — knowledge representation track (deadline ~Aug/Sep)
- **ACL** — knowledge-grounded QA angle (deadline ~January)
- **ICLR** — novel architectures (deadline ~Sep/Oct)

### Journals
- **JAIR** (Journal of AI Research) — open access, unconventional approaches welcome
- **Artificial Intelligence** (Elsevier) — oldest AI journal, architecture-level work

## Funding Sources

### Government
- **NSF** — Small Grants for Exploratory Research (SGER), AI Institute programs
- **DARPA** — Machine Common Sense, Lifelong Learning Machines programs

### Industry
- **Amazon Science** — research awards for novel approaches
- **Google Research Scholar Program** — early-career/independent researchers

### AI Safety / Interpretability
- **Open Philanthropy** — funds interpretable AI research
- **Anthropic Grants** — AI safety research
- **MIRI-adjacent organizations** — inspectable AI systems

## Key Pitch Angle

**NOT:** "I beat a benchmark"

**YES:** "I built a system where every wrong answer is diagnosable and fixable without retraining"

This is an **AI Safety / Interpretability** argument:
- Every fact is inspectable in SQLite
- Every error traces to a specific missing fact or miscalibration
- Fixes are surgical (edit one line) not catastrophic (retrain everything)
- Knowledge is separated from reasoning — you can audit what it knows
- The system honestly admits ignorance instead of hallucinating

## Results to Cite
- 89% precision with abstention (103 facts, 3B cortex)
- 82% precision with double abstention (Sara + cortex both confident)
- 0-parameter knowledge store outperforms 1.3B parameter model
- Custom 100M cortex beats 3B model without Sara (36% vs 33%)
- Full 310-question benchmark with coverage/precision tradeoff

## Immediate Next Step
Write proper paper formatted for specific venue submission.
