"""sara-build-engine — compile the optional C++ wavefront accelerator.

The pure-Python recognizer always works. This command builds the optional
``sara_engine.so`` shared library next to the installed package so that
:class:`sara_brain.core.fast_recognizer.FastRecognizer` (and the
``get_recognizer`` factory) can use the faster C++ propagation path.

Usage:
    sara-build-engine                 # build with the default compiler
    sara-build-engine --compiler clang++
    sara-build-engine --force         # rebuild even if already present
    sara-build-engine --check         # report status without building
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from .fast_recognizer import engine_available, engine_library_path

_SOURCE_FILENAME = "engine.cpp"
# Candidate compilers tried in order when --compiler is not given.
_DEFAULT_COMPILERS = ("g++", "clang++", "c++")
_CXXFLAGS = ["-O3", "-fPIC", "-std=c++17"]
_LDFLAGS = ["-shared"]


def _source_path() -> str:
    return os.path.join(os.path.dirname(__file__), _SOURCE_FILENAME)


def _find_compiler(preferred: str | None) -> str | None:
    if preferred:
        return shutil.which(preferred)
    for candidate in _DEFAULT_COMPILERS:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def build_engine(compiler: str | None = None, force: bool = False) -> str:
    """Compile ``engine.cpp`` into ``sara_engine.so`` beside the package.

    Returns the path to the compiled library on success. Raises
    ``RuntimeError`` if no compiler is available or compilation fails,
    and ``FileNotFoundError`` if the C++ source was not packaged.
    """
    src = _source_path()
    out = engine_library_path()

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"C++ engine source not found at {src}. The source may not have "
            "been included in this install."
        )

    if os.path.exists(out) and not force:
        return out

    cc = _find_compiler(compiler)
    if cc is None:
        tried = compiler or ", ".join(_DEFAULT_COMPILERS)
        raise RuntimeError(
            f"No C++ compiler found (looked for: {tried}). Install a C++17 "
            "compiler (e.g. g++ or clang++) and try again. The pure-Python "
            "recognizer continues to work without this build."
        )

    cmd = [cc, *_CXXFLAGS, src, "-o", out, *_LDFLAGS]
    print(f"[sara-build-engine] {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "C++ engine compilation failed:\n" + (proc.stderr or proc.stdout)
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compile the optional Sara C++ wavefront accelerator."
    )
    ap.add_argument(
        "--compiler",
        default=None,
        help="C++ compiler to use (default: autodetect g++/clang++/c++)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the engine already exists",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Report whether the engine is built, without building",
    )
    args = ap.parse_args()

    if args.check:
        if engine_available():
            print(f"C++ engine is built and loadable: {engine_library_path()}")
            return 0
        print("C++ engine is NOT built. Pure-Python recognizer is in use.")
        print("Run 'sara-build-engine' to build the accelerator.")
        return 1

    try:
        out = build_engine(compiler=args.compiler, force=args.force)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[sara-build-engine] {e}", file=sys.stderr)
        return 1

    if engine_available():
        print(f"[sara-build-engine] success: {out}")
        return 0
    print(
        f"[sara-build-engine] built {out} but it did not load cleanly.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
