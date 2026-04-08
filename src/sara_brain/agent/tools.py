"""Tool definitions and dispatch for the agent loop.

Two categories:
1. Brain tools (read-only) — LLM queries Sara's knowledge
2. Action tools — file ops, code execution, shell commands
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .bridge import AgentBridge
from .sandbox import Sandbox, ExecutionResult


# ── Tool Definitions (OpenAI function-calling format) ──

BRAIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "brain_query",
            "description": "Query Sara Brain for everything she knows about a topic. Returns paths leading to and from the concept.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to query (e.g., 'python', 'flask', 'testing')",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brain_recognize",
            "description": "Give Sara properties and see what she recognizes. Uses parallel wavefront propagation to find concepts with converging paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "string",
                        "description": "Comma-separated properties (e.g., 'red, round, sweet')",
                    }
                },
                "required": ["inputs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brain_context",
            "description": "Search Sara Brain for knowledge relevant to keywords. Use this before taking action to check if Sara has relevant guidance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Space-separated keywords to search for",
                    }
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brain_summarize",
            "description": "Get a complete summary of everything Sara knows about a topic, including similar concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to summarize",
                    }
                },
                "required": ["topic"],
            },
        },
    },
]

BRAIN_CLARIFY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "brain_did_you_mean",
            "description": "Check if a term has close matches in Sara Brain. Use this when a query returns no results — it may be a misspelling. Returns candidate matches with descriptions so you can ask the user 'did you mean X?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "The term to check for close matches",
                    }
                },
                "required": ["term"],
            },
        },
    },
]

VOICE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "voice_listen",
            "description": "Record audio from the microphone and transcribe it to text using local speech recognition (whisper.cpp). Use this when the user wants to speak instead of type. Returns the transcribed text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "Recording duration in seconds (default: 5)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "voice_transcribe",
            "description": "Transcribe an existing audio file to text using local speech recognition (whisper.cpp). Supports WAV, MP3, and other common formats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    }
                },
                "required": ["path"],
            },
        },
    },
]

BRAIN_MANAGEMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "brain_import",
            "description": "Import a JSON brain export file into Sara Brain. Use this when asked to load, import, or ingest a .json brain file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the JSON brain export file",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brain_ingest",
            "description": "Ingest a document into Sara Brain. Sara reads the document, extracts facts, learns them as paths, and reports what she understood. Works with local files (.txt, .md, .html) or URLs (http/https).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "File path or URL to ingest (e.g., '/path/to/doc.md' or 'https://en.wikipedia.org/wiki/RNA')",
                    }
                },
                "required": ["source"],
            },
        },
    },
]

ACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents. For large files, use offset and limit to read specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute or relative to working directory)",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based, optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (optional, default: 500)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: working directory)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (default: false)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (default: working directory)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', '*.txt')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": "Search file contents using a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter (e.g., '*.py')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code in a sandboxed subprocess. Returns stdout, stderr, and return code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": "Execute a shell command. Returns stdout, stderr, and return code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def get_tool_definitions() -> list[dict]:
    """Return all tool definitions for the Ollama chat API."""
    # Only include voice tools if whisper.cpp is available
    from ..nlp.speech import is_available as _voice_available
    voice = VOICE_TOOLS if _voice_available() else []
    return BRAIN_TOOLS + BRAIN_CLARIFY_TOOLS + BRAIN_MANAGEMENT_TOOLS + voice + ACTION_TOOLS


# ── Dispatch ──


def _get_arg(arguments: dict, *keys: str) -> str:
    """Get an argument by trying multiple key names.

    Small models don't always use the exact parameter name from the schema.
    Falls back to the first string value in the dict if no key matches.
    """
    for key in keys:
        if key in arguments:
            return arguments[key]
    # Fallback: return the first string-like value
    for v in arguments.values():
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
    return str(next(iter(arguments.values()), ""))


def dispatch(
    tool_name: str,
    arguments: dict,
    bridge: AgentBridge,
    sandbox: Sandbox,
    cwd: str,
) -> str:
    """Route a tool call to the correct handler. Returns string result."""
    # Brain tools — use _get_arg for flexible argument matching
    if tool_name == "brain_query":
        return bridge.query(_get_arg(arguments, "topic", "query", "subject", "keywords"))
    if tool_name == "brain_recognize":
        return bridge.recognize(_get_arg(arguments, "inputs", "properties", "keywords"))
    if tool_name == "brain_context":
        return bridge.context(_get_arg(arguments, "keywords", "query", "topic", "context"))
    if tool_name == "brain_summarize":
        return bridge.summarize(_get_arg(arguments, "topic", "query", "subject", "keywords"))
    if tool_name == "brain_did_you_mean":
        return bridge.did_you_mean(_get_arg(arguments, "term", "query", "word"))
    if tool_name == "brain_import":
        return bridge.import_brain(_get_arg(arguments, "path", "file", "file_path"))
    if tool_name == "brain_ingest":
        return bridge.ingest(_get_arg(arguments, "source", "path", "url", "file"))

    # Voice tools
    if tool_name == "voice_listen":
        from ..nlp.speech import record_and_transcribe
        duration = float(arguments.get("duration", 5))
        try:
            return record_and_transcribe(duration=duration)
        except Exception as e:
            return f"Voice error: {e}"
    if tool_name == "voice_transcribe":
        from ..nlp.speech import transcribe
        try:
            return transcribe(_get_arg(arguments, "path", "file"))
        except Exception as e:
            return f"Transcribe error: {e}"

    # Action tools
    if tool_name == "read_file":
        return _read_file(arguments, cwd)
    if tool_name == "write_file":
        return _write_file(arguments, cwd)
    if tool_name == "list_directory":
        return _list_directory(arguments, cwd)
    if tool_name == "search_files":
        return _search_files(arguments, cwd)
    if tool_name == "search_content":
        return _search_content(arguments, cwd)
    if tool_name == "execute_python":
        return _execute_python(arguments, sandbox)
    if tool_name == "shell_command":
        return _execute_shell(arguments, sandbox)

    return f"Unknown tool: {tool_name}"


# ── Action tool implementations ──


def _resolve_path(path_str: str, cwd: str) -> Path:
    """Resolve a path relative to working directory."""
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


def _read_file(args: dict, cwd: str) -> str:
    path = _resolve_path(args["path"], cwd)
    if not path.is_file():
        return f"File not found: {path}"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        offset = args.get("offset", 0)
        limit = args.get("limit", 500)
        selected = lines[offset : offset + limit]
        numbered = [
            f"{i + offset + 1:4d} | {line}"
            for i, line in enumerate(selected)
        ]
        header = f"File: {path} ({len(lines)} lines total)"
        if offset > 0 or len(lines) > offset + limit:
            header += f", showing lines {offset + 1}-{min(offset + limit, len(lines))}"
        return header + "\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading {path}: {e}"


def _write_file(args: dict, cwd: str) -> str:
    path = _resolve_path(args["path"], cwd)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Written: {path} ({len(args['content'])} bytes)"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _list_directory(args: dict, cwd: str) -> str:
    path = _resolve_path(args.get("path", "."), cwd)
    if not path.is_dir():
        return f"Not a directory: {path}"
    recursive = args.get("recursive", False)
    try:
        entries = []
        if recursive:
            for p in sorted(path.rglob("*")):
                if any(part.startswith(".") for part in p.relative_to(path).parts):
                    continue
                rel = p.relative_to(path)
                marker = "/" if p.is_dir() else ""
                entries.append(f"  {rel}{marker}")
        else:
            for p in sorted(path.iterdir()):
                if p.name.startswith("."):
                    continue
                marker = "/" if p.is_dir() else ""
                entries.append(f"  {p.name}{marker}")
        if not entries:
            return f"Empty directory: {path}"
        return f"Directory: {path}\n" + "\n".join(entries[:200])
    except Exception as e:
        return f"Error listing {path}: {e}"


def _search_files(args: dict, cwd: str) -> str:
    directory = _resolve_path(args.get("directory", "."), cwd)
    pattern = args["pattern"]
    try:
        matches = sorted(directory.glob(pattern))
        matches = [m for m in matches if not any(
            part.startswith(".") for part in m.relative_to(directory).parts
        )]
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        lines = [f"Found {len(matches)} file(s) matching '{pattern}':"]
        for m in matches[:100]:
            lines.append(f"  {m.relative_to(directory)}")
        if len(matches) > 100:
            lines.append(f"  ... and {len(matches) - 100} more")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching: {e}"


def _search_content(args: dict, cwd: str) -> str:
    directory = _resolve_path(args.get("directory", "."), cwd)
    pattern = args["pattern"]
    file_glob = args.get("glob", "**/*")
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []
    try:
        for path in sorted(directory.glob(file_glob)):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(directory).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = path.relative_to(directory)
                        results.append(f"  {rel}:{i}: {line.strip()}")
            except (OSError, UnicodeDecodeError):
                continue

        if not results:
            return f"No matches for '{pattern}' in {directory}"
        lines = [f"Found {len(results)} match(es):"]
        lines.extend(results[:100])
        if len(results) > 100:
            lines.append(f"  ... and {len(results) - 100} more")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching content: {e}"


def _execute_python(args: dict, sandbox: Sandbox) -> str:
    result = sandbox.execute_python(args["code"])
    return _format_execution(result, "Python")


def _execute_shell(args: dict, sandbox: Sandbox) -> str:
    result = sandbox.execute_shell(args["command"])
    return _format_execution(result, "Shell")


def _format_execution(result: ExecutionResult, label: str) -> str:
    lines = [f"{label} execution (return code: {result.return_code})"]
    if result.timed_out:
        lines.append(f"TIMED OUT: {result.stderr}")
    else:
        if result.stdout:
            lines.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            lines.append(f"stderr:\n{result.stderr}")
    return "\n".join(lines)
