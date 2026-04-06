"""System prompt construction — injects Sara's knowledge and role."""

from __future__ import annotations

from .bridge import AgentBridge


def build_system_prompt(bridge: AgentBridge, cwd: str, user_input: str = "") -> str:
    """Build the system prompt for the LLM.

    Designed to be concrete and directive so that even small models
    understand their role and actually use tools instead of describing them.

    If user_input is provided, Sara's relevant knowledge is pre-injected
    so the LLM doesn't need to call brain_context explicitly.
    """
    brain_summary = bridge.brain_summary()

    # Pre-inject relevant knowledge from Sara based on user's input
    relevant_knowledge = ""
    if user_input:
        context = bridge.context(user_input)
        if not context.startswith("Sara has no knowledge"):
            # Truncate to first 20 facts to stay within context limits
            lines = context.split("\n")
            if len(lines) > 22:
                lines = lines[:22] + [f"  ... and {len(lines) - 22} more facts"]
            relevant_knowledge = "\n".join(lines)

    return f"""\
You are an assistant with tools. You MUST use your tools to complete tasks.
NEVER describe what tool you would call — actually call it.
NEVER output JSON tool calls as text — use the tool calling mechanism.

When the user says "read a file", call the read_file tool immediately.
When the user says "run code", call execute_python immediately.
When the user asks a question, call brain_context first to check what Sara knows.

## How You Work
You are connected to Sara Brain, a persistent knowledge database.
- You do all the work: reading files, writing code, running commands.
- Sara Brain remembers things across sessions. You check her knowledge before acting.
- After completing an action, call brain_observe to tell Sara what happened.
- If brain_validate returns a CORRECTION, follow it — Sara knows the user's preferences.

## Sara's Current Knowledge
{brain_summary}
{f"""
## Sara's Knowledge Relevant to This Query
{relevant_knowledge}

Use the facts above when responding. This is what Sara remembers.""" if relevant_knowledge else ""}

## CRITICAL RULES
1. When asked to read something, call read_file. Do not explain how to read it.
2. When asked to do something, use tools to do it. Do not explain steps.
3. Always check brain_context before writing code or making decisions.
4. After completing a task, call brain_observe with a short fact about what happened.
5. When asked "what do you know/remember", call brain_context or brain_query with relevant keywords. Sara's database IS your memory. Do not answer from your own context.
6. Be concise. Do the work, then report what you did.
7. Do not act beyond what the user asks.
8. When summarizing a document, state facts directly: "X is Y", not "X appears to be Y".

## Working Directory
{cwd}\
"""
