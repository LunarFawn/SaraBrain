#!/bin/bash
#
# Sara Brain Installer — macOS & Linux
#
# Detects your OS and installs everything needed:
#   - Python 3.11+ (if missing)
#   - Ollama + a language model
#   - whisper.cpp + speech model (optional)
#   - sox for audio recording (optional)
#   - Sara Brain Python package
#
# Usage: curl -sSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install-sara.sh | bash
#    or: ./install-sara.sh
#
# No internet required after installation. No GPU required. No cloud.
# Just a humble PC.

set -e

OS="$(uname)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || pwd)"

if [[ "$OS" == "Darwin" ]]; then
    if [[ -f "$SCRIPT_DIR/install.sh" ]]; then
        exec bash "$SCRIPT_DIR/install.sh" "$@"
    else
        exec bash <(curl -fsSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install.sh) "$@"
    fi
elif [[ "$OS" == "Linux" ]]; then
    if [[ -f "$SCRIPT_DIR/install-linux.sh" ]]; then
        exec bash "$SCRIPT_DIR/install-linux.sh" "$@"
    else
        exec bash <(curl -fsSL https://raw.githubusercontent.com/LunarFawn/SaraBrain/main/install-linux.sh) "$@"
    fi
else
    echo "Unsupported OS: $OS"
    echo "Sara Brain currently supports macOS and Linux."
    exit 1
fi
