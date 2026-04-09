#!/bin/bash
#
# Sara Brain Offline Bundle Builder
#
# Creates a self-contained tarball with everything needed to run
# Sara Brain on a machine with NO internet. Just extract and run.
#
# Includes:
#   - Ollama binary
#   - Language model (qwen2.5:3b default)
#   - whisper.cpp binary + model (optional)
#   - Python virtual environment with Sara Brain
#   - Launch scripts
#
# Usage: ./bundle.sh [--with-voice] [--model MODEL]
# Output: sara-brain-bundle-<arch>.tar.gz
#
# Then on target machine:
#   tar xzf sara-brain-bundle-*.tar.gz
#   cd sara-brain
#   ./start.sh

set -e

# ── Parse args ──
WITH_VOICE=false
MODEL="qwen2.5:3b"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-voice) WITH_VOICE=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[info]${NC}  $1"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $1"; }
fail()  { echo -e "${RED}[fail]${NC}  $1"; }

OS="$(uname)"
ARCH="$(uname -m)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║     Sara Brain Bundle Builder        ║"
echo "  ║   Creating offline install package   ║"
echo "  ╚══════════════════════════════════════╝"
echo ""
info "OS: $OS  Arch: $ARCH  Model: $MODEL  Voice: $WITH_VOICE"
echo ""

# ── Create bundle directory ──
BUNDLE_DIR=$(mktemp -d)/sara-brain
mkdir -p "$BUNDLE_DIR"/{bin,models,whisper,python}

# ── Find Python ──
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 11 ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    fail "Python 3.11+ required to build bundle"
    exit 1
fi
ok "Python: $($PYTHON --version)"

# ── Bundle Ollama binary ──
info "Bundling Ollama..."
OLLAMA_BIN=$(which ollama 2>/dev/null)
if [[ -z "$OLLAMA_BIN" ]]; then
    fail "Ollama not installed. Run install-sara.sh first."
    exit 1
fi
cp "$OLLAMA_BIN" "$BUNDLE_DIR/bin/ollama"
ok "Ollama binary copied"

# ── Bundle the model ──
info "Bundling model: $MODEL"
# Ollama stores models in ~/.ollama/models
OLLAMA_HOME="${OLLAMA_MODELS:-$HOME/.ollama}"
if [[ -d "$OLLAMA_HOME/models" ]]; then
    cp -r "$OLLAMA_HOME/models" "$BUNDLE_DIR/models/"
    ok "Ollama models copied"
else
    warn "Ollama models directory not found at $OLLAMA_HOME/models"
    warn "Make sure you've pulled the model: ollama pull $MODEL"
fi

# ── Bundle whisper.cpp (optional) ──
if [[ "$WITH_VOICE" == "true" ]]; then
    info "Bundling whisper.cpp..."
    WHISPER_BIN=$(which whisper-cpp 2>/dev/null)
    if [[ -n "$WHISPER_BIN" ]]; then
        cp "$WHISPER_BIN" "$BUNDLE_DIR/bin/whisper-cpp"
        ok "whisper.cpp binary copied"
    else
        warn "whisper-cpp not found — voice will not be available"
    fi

    # Bundle whisper model
    for search_dir in "$HOME/.sara_brain/whisper" "/opt/homebrew/share/whisper" "/usr/local/share/whisper"; do
        if ls "$search_dir"/ggml-*.bin &>/dev/null 2>&1; then
            cp "$search_dir"/ggml-*.bin "$BUNDLE_DIR/whisper/"
            ok "Whisper model copied"
            break
        fi
    done

    # Bundle sox if available
    SOX_BIN=$(which sox 2>/dev/null)
    REC_BIN=$(which rec 2>/dev/null)
    if [[ -n "$SOX_BIN" ]]; then
        cp "$SOX_BIN" "$BUNDLE_DIR/bin/sox"
        # rec is usually a symlink to sox
        ln -sf sox "$BUNDLE_DIR/bin/rec"
        ok "sox copied"
    fi
fi

# ── Bundle Sara Brain Python environment ──
info "Creating bundled Python environment..."
$PYTHON -m venv "$BUNDLE_DIR/python/venv"
source "$BUNDLE_DIR/python/venv/bin/activate"

if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    pip install "$SCRIPT_DIR" --quiet
else
    pip install "git+https://github.com/LunarFawn/SaraBrain.git" --quiet
fi
deactivate
ok "Sara Brain Python environment bundled"

# ── Create launch scripts ──
info "Creating launch scripts..."

# Start script — launches everything
cat > "$BUNDLE_DIR/start.sh" << 'STARTEOF'
#!/bin/bash
#
# Sara Brain — Offline Launcher
# Just run this. No internet needed.
#

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colors ──
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${GREEN}  Sara Brain — Starting...${NC}"
echo ""

# Set up paths
export PATH="$BUNDLE/bin:$PATH"
export OLLAMA_MODELS="$BUNDLE/models"

# Start Ollama in background with bundled models
echo -e "${BLUE}[info]${NC}  Starting Ollama..."
"$BUNDLE/bin/ollama" serve &>/dev/null &
OLLAMA_PID=$!
sleep 3

# Verify Ollama is running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Error: Could not start Ollama"
    exit 1
fi
echo -e "${GREEN}[ok]${NC}    Ollama running (PID: $OLLAMA_PID)"

# Detect model
MODEL=$(curl -s http://localhost:11434/api/tags | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = data.get('models', [])
print(models[0]['name'] if models else '')
" 2>/dev/null)

if [[ -z "$MODEL" ]]; then
    echo "Error: No models found in bundle"
    kill $OLLAMA_PID 2>/dev/null
    exit 1
fi
echo -e "${GREEN}[ok]${NC}    Model: $MODEL"

# Activate Python environment
source "$BUNDLE/python/venv/bin/activate"

# Set whisper path if bundled
if [[ -f "$BUNDLE/bin/whisper-cpp" ]]; then
    export PATH="$BUNDLE/bin:$PATH"
    WHISPER_MODEL=$(ls "$BUNDLE/whisper"/ggml-*.bin 2>/dev/null | head -1)
    if [[ -n "$WHISPER_MODEL" ]]; then
        echo -e "${GREEN}[ok]${NC}    Voice: enabled"
    fi
fi

# Get brain stats
DB_PATH="${SARA_DB:-$HOME/.sara_brain/sara.db}"
echo -e "${GREEN}[ok]${NC}    Database: $DB_PATH"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║           Sara Brain Ready           ║"
echo "  ║                                      ║"
echo "  ║   No internet. No cloud. No GPU.     ║"
echo "  ║   Just this machine.                 ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Launch sara-agent
sara-agent --model "$MODEL" "$@"

# Cleanup
kill $OLLAMA_PID 2>/dev/null
echo ""
echo "  Sara is asleep. Brain saved."
STARTEOF
chmod +x "$BUNDLE_DIR/start.sh"

# Stop script
cat > "$BUNDLE_DIR/stop.sh" << 'STOPEOF'
#!/bin/bash
pkill -f "ollama serve" 2>/dev/null
echo "Sara Brain stopped."
STOPEOF
chmod +x "$BUNDLE_DIR/stop.sh"

# REPL script (non-agent mode)
cat > "$BUNDLE_DIR/repl.sh" << 'REPLEOF'
#!/bin/bash
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BUNDLE/python/venv/bin/activate"
sara "$@"
REPLEOF
chmod +x "$BUNDLE_DIR/repl.sh"

ok "Launch scripts created"

# ── Create README ──
cat > "$BUNDLE_DIR/README.txt" << 'READMEEOF'
Sara Brain — Offline Bundle
============================

This is a self-contained Sara Brain installation.
No internet required. No GPU required. No cloud.

QUICK START:
  ./start.sh

COMMANDS:
  ./start.sh      — Launch Sara Agent (LLM + Brain)
  ./repl.sh       — Launch REPL only (no LLM)
  ./stop.sh       — Stop background services

CUSTOM DATABASE:
  SARA_DB=/path/to/brain.db ./start.sh

PAPER:
  https://zenodo.org/records/19441821

LICENSE:
  CC BY-NC-ND 4.0

READMEEOF

# ── Create tarball ──
echo ""
info "Creating archive..."
ARCHIVE_NAME="sara-brain-bundle-${OS,,}-${ARCH}.tar.gz"
PARENT_DIR=$(dirname "$BUNDLE_DIR")
tar czf "$SCRIPT_DIR/$ARCHIVE_NAME" -C "$PARENT_DIR" sara-brain
rm -rf "$PARENT_DIR"

ARCHIVE_SIZE=$(du -h "$SCRIPT_DIR/$ARCHIVE_NAME" | cut -f1)

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║         Bundle Complete              ║"
echo "  ╚══════════════════════════════════════╝"
echo ""
echo "  File: $ARCHIVE_NAME"
echo "  Size: $ARCHIVE_SIZE"
echo ""
echo "  To deploy on any $OS $ARCH machine:"
echo "    scp $ARCHIVE_NAME user@target:~/"
echo "    ssh user@target"
echo "    tar xzf $ARCHIVE_NAME"
echo "    cd sara-brain"
echo "    ./start.sh"
echo ""
echo "  No internet needed on the target machine."
echo ""
