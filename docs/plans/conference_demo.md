# Plan: Live Conference Demo

**Goal:** Teach Sara Brain a novel topic on stage, demo it answering
questions on a $59 Arduino Uno Q, show side-by-side vs a 1B model
confabulating. All live, no internet, no datacenter.

---

## The Demo Script (5 minutes on stage)

1. "Give me a topic." (audience picks something obscure)
2. Type 5-10 facts into Sara. (visible on screen, takes 30 seconds)
3. Ask Sara a question about it. (correct answer, traced to paths)
4. Ask the same question to a 1B model. (confabulation)
5. Show Sara's path trace. (every claim traceable to what was just taught)
6. "This board costs $59. The model that just failed cost millions to train."

## What Needs to Work

| Component | Status | What's needed |
|-----------|--------|---------------|
| sara-teach CLI | ✅ exists | Polish for speed |
| Wavefront propagation | ✅ works | Tested on small brains |
| Wavefront renderer | ✅ works | Source-text output |
| Copy model inference | ✅ 95% accuracy | Wire into CLI |
| Side-by-side comparison | ⬜ | Build demo script |
| ARM/Pi/Uno Q deployment | ⬜ | Test on ARM hardware |
| Graceful "I don't know" | ⬜ | Add to copy model |
| Single demo command | ⬜ | Wire everything together |
| Ollama on ARM (for 1B comparison) | ⬜ | Test llama3.2:1b on Uno Q |

## Implementation Steps

### Step 1: Wire the demo pipeline

Build `sara-demo` CLI that does:
```bash
sara-demo ask "What is X?" --brain my.db --compare llama3.2:1b
```

Output:
```
Sara Brain says: [answer from copy model]
  Path trace: fact1 → fact2 → conclusion

llama3.2:1b says: [confabulated answer]
  Source: training weights (not inspectable)
```

### Step 2: Fast teaching mode

```bash
sara-demo teach --brain my.db
> the molecular snare detects target molecules
  ✓ taught (1ms)
> the molecular snare uses conformational change
  ✓ taught (1ms)
> [Ctrl-D to finish]
```

### Step 3: ARM deployment

- Test full pipeline on Raspberry Pi 4 (same ARM arch as Uno Q)
- Package as a single install: `pip install sara-brain`
- Verify latency: teach → ask should be < 2 seconds total

### Step 4: Fallback handling

When wavefront finds nothing:
```
Sara Brain says: I don't have enough information about X.
  (Teach me with: sara-demo teach)
```

### Step 5: Polish

- Big visible terminal font for stage
- Color coding: green for Sara's answer, red for confabulation
- Path trace formatted as readable chain
- Timing displayed: "Sara: 0.3s | 1B model: 1.2s"

## Hardware Kit for Conference

- Arduino Uno Q (4GB, $59) OR Raspberry Pi 4 (4GB, $55)
- USB-C cable + power
- HDMI to projector
- Keyboard (for teaching)
- Pre-loaded: Sara Brain, copy model, Ollama + llama3.2:1b

Total hardware cost: ~$60

## Timeline

| Task | Time estimate |
|------|---------------|
| Wire demo CLI | 2-3 hours |
| Test on ARM | 1-2 hours |
| Polish + rehearse | 2-3 hours |
| **Total** | **~1 day of work** |

## Risk Mitigation

- **What if the copy model gets it wrong on stage?**
  Pre-test with the exact topic before going live. The 95% accuracy
  means 1-in-20 chance of a miss. Have a backup topic ready.

- **What if Ollama is too slow on ARM for the 1B comparison?**
  Pre-run the 1B answer and show it cached. Or run the 1B on a
  phone/laptop as the "datacenter" comparison.

- **What if the audience picks a topic Sara can't handle?**
  The demo works with ANY topic because you teach it on the spot.
  If the facts are taught correctly, Sara answers correctly. The
  risk is teaching bad facts — practice the teaching flow.
