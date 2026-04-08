#!/bin/bash
#
# Sara Brain Installer — macOS
#
# Installs everything needed to run Sara Brain locally:
#   - Homebrew (if missing)
#   - Python 3.11+ (if missing)
#   - Ollama + a language model
#   - whisper.cpp + speech model (optional)
#   - sox for audio recording (optional)
#   - Sara Brain Python package
#
# Usage: curl -sSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install.sh | bash
#    or: ./install.sh
#
# No internet required after installation. No GPU required. No cloud.
# Just a humble PC.

set -e

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[info]${NC}  $1"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $1"; }
fail()  { echo -e "${RED}[fail]${NC}  $1"; }

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║         Sara Brain Installer         ║"
echo "  ║   Path-of-Thought Knowledge System   ║"
echo "  ║                                      ║"
echo "  ║   No cloud. No GPU. Just a laptop.   ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── Check macOS ──
if [[ "$(uname)" != "Darwin" ]]; then
    fail "This installer is for macOS. Linux support coming soon."
    exit 1
fi

# ── Homebrew ──
if command -v brew &>/dev/null; then
    ok "Homebrew found"
else
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add to path for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    ok "Homebrew installed"
fi

# ── Python 3.11+ ──
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

if [[ -n "$PYTHON" ]]; then
    ok "Python found: $($PYTHON --version)"
else
    info "Installing Python 3.12..."
    brew install python@3.12
    PYTHON="python3.12"
    ok "Python 3.12 installed"
fi

# ── Ollama ──
if command -v ollama &>/dev/null; then
    ok "Ollama found"
else
    info "Installing Ollama..."
    brew install ollama
    ok "Ollama installed"
fi

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    ok "Ollama is running"
else
    info "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama started"
    else
        warn "Could not start Ollama. Run 'ollama serve' manually after install."
    fi
fi

# ── Pull a language model ──
echo ""
info "Checking for language models..."
MODEL_COUNT=$(curl -s http://localhost:11434/api/tags 2>/dev/null | $PYTHON -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    print(len(models))
    for m in models:
        print(m.get('name', ''), file=sys.stderr)
except:
    print(0)
" 2>/dev/null)

if [[ "$MODEL_COUNT" -gt 0 ]]; then
    ok "Found $MODEL_COUNT model(s)"
else
    echo ""
    echo "  Sara needs a language model (her sensory cortex)."
    echo "  Recommended options:"
    echo ""
    echo "    1. qwen2.5:3b     — 1.9GB, fast, good for testing"
    echo "    2. llama3.2:3b    — 2.0GB, good all-around"
    echo "    3. llama3.1:8b    — 4.7GB, best quality (needs 8GB+ RAM)"
    echo ""
    read -p "  Which model? [1/2/3] (default: 1): " MODEL_CHOICE
    case "${MODEL_CHOICE:-1}" in
        1) MODEL="qwen2.5:3b" ;;
        2) MODEL="llama3.2:3b" ;;
        3) MODEL="llama3.1:8b" ;;
        *) MODEL="qwen2.5:3b" ;;
    esac
    info "Pulling $MODEL (this may take a few minutes)..."
    ollama pull "$MODEL"
    ok "$MODEL ready"
fi

# ── whisper.cpp (optional) ──
echo ""
echo "  Speech recognition lets Sara listen to voice input."
echo "  This is optional — Sara works fine with text only."
echo ""
read -p "  Install speech recognition? [y/N]: " INSTALL_VOICE
if [[ "${INSTALL_VOICE,,}" == "y" ]]; then
    if command -v whisper-cpp &>/dev/null; then
        ok "whisper.cpp found"
    else
        info "Installing whisper.cpp..."
        brew install whisper-cpp
        ok "whisper.cpp installed"
    fi

    # Download whisper model
    WHISPER_DIR="$HOME/.sara_brain/whisper"
    mkdir -p "$WHISPER_DIR"
    if ls "$WHISPER_DIR"/ggml-*.bin &>/dev/null || ls /opt/homebrew/share/whisper/ggml-*.bin &>/dev/null 2>&1; then
        ok "Whisper model found"
    else
        info "Downloading whisper base.en model (~140MB)..."
        if command -v whisper-cpp-download-ggml-model &>/dev/null; then
            whisper-cpp-download-ggml-model base.en
        else
            # Manual download
            curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" \
                -o "$WHISPER_DIR/ggml-base.en.bin"
        fi
        ok "Whisper model ready"
    fi

    # sox for audio recording
    if command -v sox &>/dev/null || command -v rec &>/dev/null; then
        ok "sox found (audio recording)"
    else
        info "Installing sox (audio recording)..."
        brew install sox
        ok "sox installed"
    fi
else
    info "Skipping speech recognition (install later with: brew install whisper-cpp sox)"
fi

# ── Sara Brain ──
echo ""
info "Installing Sara Brain..."

# Create virtual environment
SARA_DIR="$HOME/.sara_brain"
mkdir -p "$SARA_DIR"
VENV_DIR="$SARA_DIR/.venv"

if [[ -d "$VENV_DIR" ]]; then
    ok "Virtual environment exists"
else
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate and install
source "$VENV_DIR/bin/activate"

# Determine install source — local repo or pip
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    info "Installing from local source..."
    pip install -e "$SCRIPT_DIR" --quiet
else
    info "Installing from GitHub..."
    pip install "git+https://github.com/LunarFawn/SaraBrain.git" --quiet
fi
ok "Sara Brain installed"

# ── Create shell wrapper scripts ──
info "Creating command shortcuts..."

BIN_DIR="$SARA_DIR/bin"
mkdir -p "$BIN_DIR"

# sara command
cat > "$BIN_DIR/sara" << 'WRAPPER'
#!/bin/bash
source "$HOME/.sara_brain/.venv/bin/activate"
exec sara "$@"
WRAPPER
chmod +x "$BIN_DIR/sara"

# sara-agent command
cat > "$BIN_DIR/sara-agent" << 'WRAPPER'
#!/bin/bash
source "$HOME/.sara_brain/.venv/bin/activate"
exec sara-agent "$@"
WRAPPER
chmod +x "$BIN_DIR/sara-agent"

ok "Commands created in $BIN_DIR"

# ── Add to PATH ──
SHELL_RC=""
if [[ -f "$HOME/.zshrc" ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ -f "$HOME/.bashrc" ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [[ -n "$SHELL_RC" ]]; then
    if ! grep -q "sara_brain/bin" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# Sara Brain" >> "$SHELL_RC"
        echo "export PATH=\"\$HOME/.sara_brain/bin:\$PATH\"" >> "$SHELL_RC"
        ok "Added to PATH in $SHELL_RC"
    else
        ok "PATH already configured"
    fi
fi

# ── Done ──
echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║        Installation Complete         ║"
echo "  ╚══════════════════════════════════════╝"
echo ""
echo "  Commands:"
echo "    sara          — Interactive REPL"
echo "    sara-agent    — Ollama agent (LLM + Sara Brain)"
echo ""
echo "  Quick start:"
echo "    sara-agent"
echo "    you> teach me about apples"
echo ""
echo "  Paper: https://zenodo.org/records/19441821"
echo ""
echo "  Restart your terminal or run:"
echo "    export PATH=\"\$HOME/.sara_brain/bin:\$PATH\""
echo ""
