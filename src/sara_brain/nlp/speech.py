"""Speech recognition — local voice-to-text via whisper.cpp.

Sara's ears. Converts spoken audio to text that feeds through the
fuzzy neuron resolver. Runs entirely locally, no cloud, no GPU required.

Requires whisper.cpp installed: brew install whisper-cpp
And a model downloaded: whisper-cpp-download-ggml-model base.en
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


# Default whisper.cpp model path
_DEFAULT_MODEL_DIR = Path.home() / ".sara_brain" / "whisper"
_DEFAULT_MODEL = "ggml-base.en.bin"


def is_available() -> bool:
    """Check if whisper.cpp is installed and a model exists."""
    return (
        shutil.which("whisper-cpp") is not None
        and get_model_path() is not None
    )


def get_model_path() -> Path | None:
    """Find the whisper model file."""
    # Check Sara's model directory first
    sara_model = _DEFAULT_MODEL_DIR / _DEFAULT_MODEL
    if sara_model.is_file():
        return sara_model

    # Check common whisper.cpp model locations
    common_paths = [
        Path.home() / ".local" / "share" / "whisper" / _DEFAULT_MODEL,
        Path("/usr/local/share/whisper") / _DEFAULT_MODEL,
        Path("/opt/homebrew/share/whisper") / _DEFAULT_MODEL,
    ]
    for p in common_paths:
        if p.is_file():
            return p

    # Search for any ggml model file
    for d in [_DEFAULT_MODEL_DIR] + common_paths:
        parent = d.parent
        if parent.is_dir():
            for f in parent.iterdir():
                if f.name.startswith("ggml-") and f.name.endswith(".bin"):
                    return f

    return None


def transcribe(audio_path: str, model_path: str | None = None,
               language: str = "en") -> str:
    """Transcribe an audio file to text using whisper.cpp.

    Args:
        audio_path: Path to audio file (WAV preferred, 16kHz mono).
        model_path: Path to whisper model. Auto-detected if None.
        language: Language code (default: 'en').

    Returns:
        Transcribed text string.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model_path is None:
        mp = get_model_path()
        if mp is None:
            raise RuntimeError(
                "No whisper model found. Install with:\n"
                "  brew install whisper-cpp\n"
                "  whisper-cpp-download-ggml-model base.en"
            )
        model_path = str(mp)

    # whisper.cpp needs 16kHz mono WAV — convert if needed
    wav_path = _ensure_16khz_wav(audio_path)

    try:
        result = subprocess.run(
            [
                "whisper-cpp",
                "-m", model_path,
                "-l", language,
                "-nt",  # no timestamps
                "-f", str(wav_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(f"whisper-cpp failed: {result.stderr}")

        # Clean up the output — strip whitespace and whisper artifacts
        text = result.stdout.strip()
        # Remove common whisper artifacts
        text = text.replace("[BLANK_AUDIO]", "").strip()

        return text

    finally:
        # Clean up temp file if we created one
        if str(wav_path) != str(audio_path) and wav_path.exists():
            wav_path.unlink()


def record_and_transcribe(duration: float = 5.0,
                          model_path: str | None = None) -> str:
    """Record from microphone and transcribe.

    Uses macOS 'rec' (from sox) or 'arecord' (Linux) to capture audio,
    then transcribes with whisper.cpp.

    Args:
        duration: Recording duration in seconds.
        model_path: Path to whisper model. Auto-detected if None.

    Returns:
        Transcribed text string.
    """
    # Create temp WAV file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = tmp.name

    try:
        # Try sox/rec first (macOS + Linux)
        rec_cmd = shutil.which("rec")
        if rec_cmd:
            subprocess.run(
                [rec_cmd, "-r", "16000", "-c", "1", "-b", "16",
                 tmp_path, "trim", "0", str(duration)],
                capture_output=True,
                timeout=duration + 5,
            )
        else:
            # Try arecord (Linux)
            arecord_cmd = shutil.which("arecord")
            if arecord_cmd:
                subprocess.run(
                    [arecord_cmd, "-f", "S16_LE", "-r", "16000", "-c", "1",
                     "-d", str(int(duration)), tmp_path],
                    capture_output=True,
                    timeout=duration + 5,
                )
            else:
                raise RuntimeError(
                    "No audio recorder found. Install sox:\n"
                    "  brew install sox"
                )

        return transcribe(tmp_path, model_path=model_path)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _ensure_16khz_wav(audio_path: Path) -> Path:
    """Convert audio to 16kHz mono WAV if needed."""
    # Check if it's already a valid WAV at 16kHz
    try:
        with wave.open(str(audio_path), "rb") as wf:
            if wf.getframerate() == 16000 and wf.getnchannels() == 1:
                return audio_path
    except (wave.Error, EOFError):
        pass  # Not a valid WAV, needs conversion

    # Convert using sox or ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    sox_cmd = shutil.which("sox")
    if sox_cmd:
        subprocess.run(
            [sox_cmd, str(audio_path), "-r", "16000", "-c", "1",
             "-b", "16", tmp.name],
            capture_output=True,
            timeout=30,
        )
        return Path(tmp.name)

    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        subprocess.run(
            [ffmpeg_cmd, "-i", str(audio_path), "-ar", "16000", "-ac", "1",
             "-y", tmp.name],
            capture_output=True,
            timeout=30,
        )
        return Path(tmp.name)

    raise RuntimeError(
        "Cannot convert audio format. Install sox or ffmpeg:\n"
        "  brew install sox"
    )
