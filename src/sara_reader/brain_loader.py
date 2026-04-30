"""Open a Sara brain at a given file path.

The path can be local or any path the OS resolves (network-mounted shares,
symlinks, etc.). sara_reader does not implement remote-brain protocols;
remote access is handled at the filesystem layer.
"""
from __future__ import annotations

from pathlib import Path

from sara_brain.core.brain import Brain


def load_brain(brain_path: str | Path) -> Brain:
    """Open the brain at ``brain_path`` and return a Brain instance.

    Args:
        brain_path: filesystem path to a Sara .db file. Local or
            network-mounted (NFS, SMB, fuse-mounted S3, etc.) — the
            difference is invisible at this layer.

    Returns:
        A Brain ready to be queried. The caller is responsible for
        the brain's lifecycle (typically held for the duration of a
        SaraReader instance).
    """
    p = Path(brain_path)
    if not p.exists():
        raise FileNotFoundError(f"Brain not found at {p!s}")
    return Brain(str(p))
