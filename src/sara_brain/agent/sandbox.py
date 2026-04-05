"""Code execution sandbox — subprocess with timeout.

Runs Python code or shell commands in a child process. No pip dependencies.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool


class Sandbox:
    """Execute code in a child process with timeout."""

    def __init__(self, timeout: int = 30, cwd: str | None = None) -> None:
        self.timeout = timeout
        self.cwd = cwd or str(Path.cwd())

    def execute_python(self, code: str) -> ExecutionResult:
        """Write code to a temp file, run with current Python interpreter."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.cwd
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            return self._run([sys.executable, script_path])
        finally:
            Path(script_path).unlink(missing_ok=True)

    def execute_shell(self, command: str) -> ExecutionResult:
        """Run a shell command."""
        return self._run(command, shell=True)

    def _run(
        self, cmd: str | list[str], shell: bool = False
    ) -> ExecutionResult:
        timed_out = False
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
                shell=shell,
                stdin=subprocess.DEVNULL,
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Timed out after {self.timeout}s",
                return_code=-1,
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                timed_out=False,
            )
