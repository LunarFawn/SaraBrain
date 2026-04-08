#!/bin/bash
#
# Sara Brain Installer — Linux (Debian/Ubuntu/Fedora/Arch)
#
# Installs everything needed to run Sara Brain locally:
#   - Python 3.11+ (if missing)
#   - Ollama + a language model
#   - whisper.cpp + speech model (optional)
#   - sox for audio recording (optional)
#   - Sara Brain Python package
#
# Usage: curl -sSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install-linux.sh | bash
#    or: ./install-linux.sh
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

# ── Check Linux ──
if [[ "$(uname)" != "Linux" ]]; then
    fail "This installer is for Linux. Use install.sh for macOS."
    exit 1
fi

# ── Detect package manager ──
PKG=""
if command -v apt-get &>/dev/null; then
    PKG="apt"
elif command -v dnf &>/dev/null; then
    PKG="dnf"
elif command -v pacman &>/dev/null; then
    PKG="pacman"
elif command -v zypper &>/dev/null; then
    PKG="zypper"
else
    fail "Could not detect package manager (apt/dnf/pacman/zypper)."
    fail "Install Python 3.11+, curl, and git manually, then re-run."
    exit 1
fi
ok "Package manager: $PKG"

# ── Install system dependencies ──
info "Checking system dependencies..."
NEED_INSTALL=()

for cmd in curl git; do
    if ! command -v "$cmd" &>/dev/null; then
        NEED_INSTALL+=("$cmd")
    fi
done

if [[ ${#NEED_INSTALL[@]} -gt 0 ]]; then
    info "Installing: ${NEED_INSTALL[*]}"
    case "$PKG" in
        apt)    sudo apt-get update -qq && sudo apt-get install -y -qq "${NEED_INSTALL[@]}" ;;
        dnf)    sudo dnf install -y -q "${NEED_INSTALL[@]}" ;;
        pacman) sudo pacman -S --noconfirm "${NEED_INSTALL[@]}" ;;
        zypper) sudo zypper install -y "${NEED_INSTALL[@]}" ;;
    esac
fi
ok "System dependencies ready"

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
    case "$PKG" in
        apt)
            sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev 2>/dev/null \
                || sudo apt-get install -y -qq python3 python3-venv python3-dev
            ;;
        dnf)    sudo dnf install -y -q python3.12 python3.12-devel 2>/dev/null \
                || sudo dnf install -y -q python3 python3-devel ;;
        pacman) sudo pacman -S --noconfirm python ;;
        zypper) sudo zypper install -y python312 python312-devel 2>/dev/null \
                || sudo zypper install -y python3 python3-devel ;;
    esac
    # Re-detect
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
        fail "Could not install Python 3.11+. Please install manually."
        exit 1
    fi
    ok "Python installed: $($PYTHON --version)"
fi

# Check for venv module
if ! $PYTHON -m venv --help &>/dev/null; then
    info "Installing Python venv module..."
    case "$PKG" in
        apt)    sudo apt-get install -y -qq python3-venv ;;
        dnf)    : ;; # included by default
        pacman) : ;; # included by default
        zypper) : ;; # included by default
    esac
fi

# ── Ollama ──
if command -v ollama &>/dev/null; then
    ok "Ollama found"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
fi

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    ok "Ollama is running"
else
    info "Starting Ollama..."
    # Try systemd first
    if command -v systemctl &>/dev/null; then
        sudo systemctl start ollama 2>/dev/null || ollama serve &>/dev/null &
    else
        ollama serve &>/dev/null &
    fi
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
    if command -v whisper-cpp &>/dev/null || command -v main &>/dev/null; then
        ok "whisper.cpp found"
    else
        info "Building whisper.cpp from source..."
        BUILD_DIR=$(mktemp -d)
        git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "$BUILD_DIR/whisper.cpp"
        cd "$BUILD_DIR/whisper.cpp"

        # Install build deps
        case "$PKG" in
            apt)    sudo apt-get install -y -qq build-essential cmake ;;
            dnf)    sudo dnf install -y -q gcc-c++ cmake make ;;
            pacman) sudo pacman -S --noconfirm base-devel cmake ;;
            zypper) sudo zypper install -y gcc-c++ cmake make ;;
        esac

        cmake -B build
        cmake --build build --config Release -j$(nproc)
        sudo cp build/bin/whisper-cpp /usr/local/bin/ 2>/dev/null \
            || sudo cp build/bin/main /usr/local/bin/whisper-cpp
        cd -
        rm -rf "$BUILD_DIR"
        ok "whisper.cpp built and installed"
    fi

    # Download whisper model
    WHISPER_DIR="$HOME/.sara_brain/whisper"
    mkdir -p "$WHISPER_DIR"
    if ls "$WHISPER_DIR"/ggml-*.bin &>/dev/null 2>&1; then
        ok "Whisper model found"
    else
        info "Downloading whisper base.en model (~140MB)..."
        curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" \
            -o "$WHISPER_DIR/ggml-base.en.bin"
        ok "Whisper model ready"
    fi

    # sox / alsa-utils for audio recording
    if command -v sox &>/dev/null || command -v rec &>/dev/null || command -v arecord &>/dev/null; then
        ok "Audio recording tools found"
    else
        info "Installing audio recording tools..."
        case "$PKG" in
            apt)    sudo apt-get install -y -qq sox alsa-utils ;;
            dnf)    sudo dnf install -y -q sox alsa-utils ;;
            pacman) sudo pacman -S --noconfirm sox alsa-utils ;;
            zypper) sudo zypper install -y sox alsa-utils ;;
        esac
        ok "Audio tools installed"
    fi
else
    info "Skipping speech recognition"
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
if [[ -f "$HOME/.bashrc" ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ -f "$HOME/.zshrc" ]]; then
    SHELL_RC="$HOME/.zshrc"
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
