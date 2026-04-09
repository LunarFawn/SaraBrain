# v011 — Sara Brain Installation & Usage Guide

## Quick Start

One command installs everything:

```bash
curl -sSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install-sara.sh | bash
```

This auto-detects your OS (macOS or Linux) and installs:
- Python 3.11+
- Ollama + a language model
- whisper.cpp + speech model (optional)
- Sara Brain in a virtual environment

No internet required after installation. No GPU required. No cloud.

## What Gets Installed

| Component | Purpose | Size |
|-----------|---------|------|
| **Ollama** | Local LLM runtime (Sara's sensory cortex) | ~500MB |
| **qwen2.5:3b** | Default language model | ~1.9GB |
| **whisper.cpp** | Local speech recognition (optional) | ~10MB |
| **Whisper model** | Speech-to-text model (optional) | ~140MB |
| **sox** | Audio recording (optional) | ~5MB |
| **Sara Brain** | Path-of-thought knowledge system | ~1MB |

Total with voice: ~2.5GB. Without voice: ~2.4GB.

## Platform Support

### macOS

Requires Homebrew (installed automatically if missing).

```bash
# Automatic install
curl -sSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install-sara.sh | bash

# Or from a local clone
git clone https://github.com/LunarFawn/SaraBrain.git
cd SaraBrain
./install.sh
```

### Linux (Debian/Ubuntu/Fedora/Arch)

Auto-detects package manager (apt/dnf/pacman/zypper).

```bash
# Automatic install
curl -sSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install-sara.sh | bash

# Or from a local clone
git clone https://github.com/LunarFawn/SaraBrain.git
cd SaraBrain
./install-linux.sh
```

### Raspberry Pi

Works on Pi 4 or Pi 5 with 4GB+ RAM. Uses the Linux installer.

```bash
curl -sSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install-sara.sh | bash
```

Recommended model for Pi: **qwen2.5:3b** (fits in 4GB RAM).

## Offline Bundle (No Internet on Target)

For deploying to machines with no internet — rural clinics, air-gapped systems, conference demos.

### Build the bundle (on a machine with internet)

```bash
git clone https://github.com/LunarFawn/SaraBrain.git
cd SaraBrain

# Without voice
./bundle.sh

# With voice support
./bundle.sh --with-voice

# With a specific model
./bundle.sh --model llama3.1:8b --with-voice
```

This creates `sara-brain-bundle-<os>-<arch>.tar.gz`.

### Deploy (no internet required)

```bash
# Copy to target machine
scp sara-brain-bundle-*.tar.gz user@target:~/

# On the target machine
tar xzf sara-brain-bundle-*.tar.gz
cd sara-brain
./start.sh
```

That's it. No install, no internet, no dependencies. Extract and run.

### Bundle contents

```
sara-brain/
├── start.sh          # Launch Sara Agent (LLM + Brain)
├── repl.sh           # Launch REPL only (no LLM)
├── stop.sh           # Stop background services
├── README.txt
├── bin/
│   ├── ollama        # Ollama binary
│   ├── whisper-cpp   # Speech recognition (if --with-voice)
│   └── sox           # Audio recording (if --with-voice)
├── models/           # Pre-pulled Ollama models
├── whisper/          # Whisper speech model (if --with-voice)
└── python/
    └── venv/         # Python environment with Sara Brain
```

## Running Sara Brain

### Sara Agent (recommended)

Interactive agent with LLM sensory cortex + Sara Brain cerebellum.

```bash
sara-agent
```

Options:
```
sara-agent --model qwen2.5-coder:3b    # Specific model
sara-agent --db /path/to/brain.db       # Custom database
sara-agent --url http://localhost:11434  # Custom Ollama URL
sara-agent --session 2026-04-08_001     # Resume a session
```

Example session:
```
you> teach me about apples
sara> I'll teach Sara about apples.
      Learned: red → apple_color → apple

you> what do you know about apples?
sara> Sara knows:
      apple is red (from: "apples are red")
      apple is round (from: "apples are round")
      apple is sweet (from: "apples are sweet")

you> recognize red, round, sweet
sara> Recognized: apple (3 converging paths)
```

### Sara REPL

Direct brain interaction without an LLM. Useful for teaching, querying, and debugging.

```bash
sara
```

Commands:
```
teach <statement>          — Teach a fact (e.g., "teach apples are red")
recognize <inputs>         — Recognize from properties (e.g., "recognize red, round")
trace <label>              — Show all paths from a neuron
why <label>                — Show all paths leading to a neuron
similar <label>            — Find neurons sharing paths
stats                      — Brain statistics
neurons                    — List all neurons
paths                      — List all paths
perceive <image> [label]   — Perceive an image (requires LLM)
ask <question>             — Ask a natural language question (requires LLM)
llm set ollama             — Configure local Ollama LLM
quit                       — Exit
```

### Document Ingestion

Sara can learn from documents — local files or web URLs.

In the agent:
```
you> ingest this document: /path/to/medical_guide.pdf
you> ingest https://en.wikipedia.org/wiki/Insulin
```

Sara reads the document through the LLM cortex, extracts facts, learns them as neuron-chain paths, identifies unknown concepts, and reports what she understood.

### Voice Input

If whisper.cpp is installed, Sara can listen to voice input.

```
you> listen to me
[Recording for 5 seconds...]
sara> I heard: "tell me about metformin"
      Did you mean: metformin (medication) — used for type 2 diabetes?
```

Voice is processed entirely locally. No audio leaves the machine.

## Configuration

Sara Brain stores everything in `~/.sara_brain/`:

```
~/.sara_brain/
├── sara.db               # Default brain database
├── sessions/             # Conversation history
├── whisper/              # Whisper models
└── .venv/                # Python virtual environment
```

### LLM Configuration

```bash
# In the REPL
sara> llm set ollama
sara> llm status

# Or via environment
export OLLAMA_HOST=http://localhost:11434
```

### Custom Database

Each brain is a single SQLite file. You can have multiple brains:

```bash
sara --db ~/medical.db          # Medical knowledge brain
sara --db ~/engineering.db      # Engineering brain
sara-agent --db ~/clinic.db     # Clinic brain
```

## Architecture

Sara Brain is a path-of-thought knowledge system. Knowledge lives in neuron chains, not individual nodes.

```
User speaks → LLM (sensory cortex) → Structured facts → Sara Brain (cerebellum)
                                                              ↓
                                                     Neurons → Segments → Paths
                                                              ↓
                                                     Parallel wavefront recognition
                                                              ↓
                                                     Answers with traceable reasoning
```

- **LLM** = eyes, ears, mouth (replaceable, local)
- **Sara Brain** = the knowledge (persistent, traceable, never forgets)
- **SQLite** = storage (portable, zero-config)

The LLM can be swapped, upgraded, or changed without losing any knowledge. The brain persists across sessions, models, and hardware.

## Fuzzy Matching

Sara handles misspellings and word variants automatically:

- "summerians" → sumerian (edit distance)
- "sumerians" → sumerian (inflection)
- "sumer" → sumerian (prefix match)
- "writng" → writing (edit distance)
- "bier" → beer (edit distance)

When multiple candidates match, Sara asks "did you mean?" for safe disambiguation — critical for medical terms where a wrong match could be dangerous.

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | macOS 12+ or Linux | macOS 14+ or Ubuntu 22.04+ |
| Python | 3.11 | 3.12 |
| RAM | 4GB | 8GB |
| Disk | 3GB | 5GB |
| GPU | Not required | Not required |
| Internet | Only for install | Only for install |

## Troubleshooting

**Ollama not starting:**
```bash
ollama serve           # Start manually
curl localhost:11434   # Check if running
```

**No models found:**
```bash
ollama list            # List installed models
ollama pull qwen2.5:3b # Pull a model
```

**Import errors:**
```bash
# Make sure you're using the Sara Brain venv
source ~/.sara_brain/.venv/bin/activate
sara --help
```

**Voice not working:**
```bash
whisper-cpp --help     # Check if installed
ls ~/.sara_brain/whisper/  # Check for model
brew install sox       # Install audio recording (macOS)
```

## Paper

Pearl, J. (2026). Path-of-Thought: A Neuron-Chain Knowledge Representation System with Parallel Wavefront Recognition. Zenodo. https://doi.org/10.5281/zenodo.19441821

## License

CC BY-NC-ND 4.0

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v011 | 2026-04-08 | Installation guide: quick start, offline bundles, platform support, voice, fuzzy matching |
| v010 | 2026-04-06 | Sara Care: dementia assistance proof-of-concept |
| v009 | 2026-03-24 | Sara steered an LLM: QMSE quality enforcement demo |
