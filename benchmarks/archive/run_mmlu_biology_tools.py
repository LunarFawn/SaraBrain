#!/usr/bin/env python3
"""MMLU Biology benchmark — tool-calling version (the ARCHITECTURALLY CORRECT way).

The LLM answers each multiple-choice question with access to Sara's brain
via tool calls, NOT via a prompt dump. This mirrors how a real
cortex-cerebellum system is supposed to work:

  - LLM reads the question
  - LLM decides what it needs to know
  - LLM calls brain_query(topic) or brain_context(keywords)
  - Sara returns what she knows
  - LLM continues querying or gives the final answer

This tests whether Sara's knowledge actually helps a small model
when accessed properly, vs the shortcut of dumping all matched
paths into the system prompt.

Usage:
    python benchmarks/run_mmlu_biology_tools.py --baseline
    python benchmarks/run_mmlu_biology_tools.py --db bio_full.db
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.error


# Only the read-only query tools — no teaching, no action tools
BRAIN_QUERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "brain_query",
            "description": (
                "Query Sara's knowledge graph for what she knows about a "
                "topic. Returns paths leading to and from the concept. "
                "Use this for specific terms (a single concept)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Specific topic term (one concept)",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brain_context",
            "description": (
                "Search Sara's brain for knowledge relevant to keywords. "
                "Use this when you have multiple keywords or want broader context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Space-separated keywords",
                    }
                },
                "required": ["keywords"],
            },
        },
    },
]


SYSTEM_WITH_SARA = """You are a biology student taking a multiple-choice exam.

Before answering, you MUST look up the key concept in your knowledge
graph. You have two tools available:

- brain_query: {"name": "brain_query", "arguments": {"topic": "<concept>"}}
- brain_context: {"name": "brain_context", "arguments": {"keywords": "<words>"}}

Your process:
1. FIRST output a tool call as JSON (exactly the format above), nothing else
2. You will receive the knowledge back
3. Then output only the letter A, B, C, or D

Example of a correct first response:
{"name": "brain_query", "arguments": {"topic": "photosynthesis"}}

Do not answer without looking something up first. Trust the knowledge
graph. Final answer is just the letter, no other text."""


SYSTEM_BASELINE = """You are a biology expert answering a multiple-choice exam.

Answer the question with ONLY the letter (A, B, C, or D). Nothing else."""


def load_questions() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    return [
        {
            'id': i,
            'question': q['question'],
            'choices': q['choices'],
            'answer_idx': q['answer'],
        }
        for i, q in enumerate(ds)
    ]


def format_mc_prompt(question: str, choices: list[str]) -> str:
    labels = ['A', 'B', 'C', 'D']
    lines = [question, '']
    for i, choice in enumerate(choices):
        lines.append(f'{labels[i]}. {choice}')
    lines.append('')
    return '\n'.join(lines)


def ollama_chat(messages: list[dict], model: str, base_url: str,
                tools: list | None = None, max_tokens: int = 500) -> dict:
    """Call Ollama chat with tool-calling (structured OR text-parsed).

    Uses the existing sara_brain.agent.ollama.chat + extract_response
    which handles both native tool calling AND text-parsed fallback
    for small models that output JSON in their content instead of
    using the structured tools field.
    """
    from sara_brain.agent.ollama import chat, extract_response

    body = chat(base_url=base_url, model=model, messages=messages,
                tools=tools, temperature=0, max_tokens=max_tokens)
    # extract_response handles the text-parsing fallback for small models
    result = extract_response(body)
    # Return in the shape the caller expects (OpenAI-compat)
    return {
        "choices": [{
            "message": {
                "content": result.get("content"),
                "tool_calls": result.get("tool_calls"),
            }
        }]
    }


def extract_answer(text: str) -> str | None:
    text = text.strip().upper()
    if text in ("A", "B", "C", "D"):
        return text
    for char in text:
        if char in "ABCD":
            return char
    return None


def run_one_question(q: dict, model: str, base_url: str,
                     bridge=None, max_tool_rounds: int = 4) -> dict:
    """Run a single question. If bridge is provided, LLM has tool access."""
    prompt = format_mc_prompt(q['question'], q['choices'])
    use_tools = bridge is not None

    if use_tools:
        # Prepend a directive that forces tool use before answering
        user_content = (
            f"{prompt}\n\n"
            "Before answering, call brain_query on the most important "
            "biology concept in this question. You MUST call at least one "
            "tool first. After the tool result, respond with only the letter."
        )
    else:
        user_content = prompt

    messages = [
        {"role": "system",
         "content": SYSTEM_WITH_SARA if use_tools else SYSTEM_BASELINE},
        {"role": "user", "content": user_content},
    ]

    tool_calls_made = 0

    for round_num in range(max_tool_rounds + 1):
        # Only pass tools on rounds where we still allow them
        tools_this_round = BRAIN_QUERY_TOOLS if (use_tools and round_num < max_tool_rounds) else None
        response = ollama_chat(messages, model, base_url, tools=tools_this_round)
        msg = response["choices"][0]["message"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls") or []

        # If no tool calls, this is the final answer
        if not tool_calls:
            answer = extract_answer(content)
            return {"answer": answer, "raw": content[:300],
                    "tool_calls": tool_calls_made}

        # Process tool calls
        messages.append({"role": "assistant", "content": content or "",
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            tool_calls_made += 1
            name = tc.get("function", {}).get("name", "")
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            if name == "brain_query":
                result = bridge.query(args.get("topic", ""))
            elif name == "brain_context":
                result = bridge.context(args.get("keywords", ""))
            else:
                result = f"Unknown tool: {name}"

            # Cap tool result length so we don't drown the model
            if len(result) > 2000:
                result = result[:2000] + "\n[...truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })

    # Ran out of rounds — force a final answer
    messages.append({"role": "user",
                     "content": "Based on what you know, give your final answer "
                                "as just the letter A, B, C, or D."})
    response = ollama_chat(messages, model, base_url, tools=None)
    content = response["choices"][0]["message"].get("content", "")
    return {"answer": extract_answer(content), "raw": content[:300],
            "tool_calls": tool_calls_made}


def run_benchmark(questions: list[dict], model: str, base_url: str,
                  bridge=None) -> dict:
    results = {
        "mode": "sara+llm (tool-calling)" if bridge else "llm_only",
        "model": model,
        "total": len(questions),
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "total_tool_calls": 0,
        "answers": [],
    }

    bench_start = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        try:
            result = run_one_question(q, model, base_url, bridge=bridge)
            answer = result["answer"]
            tcs = result["tool_calls"]
        except Exception as e:
            answer = None
            tcs = 0
            print(f"  [{i+1}/{len(questions)}] Q{q['id']}: ERROR — {e}",
                  flush=True)

        correct_letter = ["A", "B", "C", "D"][q['answer_idx']]
        is_correct = answer == correct_letter

        if answer is None:
            results["errors"] += 1
        elif is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1
        results["total_tool_calls"] += tcs

        results["answers"].append({
            "id": q['id'],
            "correct_letter": correct_letter,
            "model_answer": answer,
            "is_correct": is_correct,
            "tool_calls": tcs,
        })

        elapsed = time.time() - q_start
        total_elapsed = time.time() - bench_start
        status = "CORRECT" if is_correct else ("ERROR" if answer is None else "WRONG")
        accuracy = results["correct"] / (i + 1) * 100
        avg = total_elapsed / (i + 1)
        remaining = avg * (len(questions) - i - 1)
        tc_info = f" {tcs}tc" if bridge else ""
        print(f"  [{i+1}/{len(questions)}] Q{q['id']}: {status} "
              f"(got {answer}, correct {correct_letter}){tc_info} "
              f"— {accuracy:.1f}% — {elapsed:.1f}s (~{remaining/60:.0f}m left)",
              flush=True)

    total_time = time.time() - bench_start
    results["accuracy"] = results["correct"] / results["total"] * 100
    results["total_time_sec"] = total_time
    return results


def print_summary(results: dict) -> None:
    print()
    print(f"  {'='*60}")
    print(f"  MMLU High School Biology — {results['mode']}")
    print(f"  Model: {results['model']}")
    print(f"  {'='*60}")
    print(f"  Total:       {results['total']}")
    print(f"  Correct:     {results['correct']} ({results['accuracy']:.1f}%)")
    print(f"  Incorrect:   {results['incorrect']}")
    print(f"  Errors:      {results['errors']}")
    if results.get('total_tool_calls'):
        print(f"  Tool calls:  {results['total_tool_calls']} total, "
              f"{results['total_tool_calls']/results['total']:.1f} avg/question")
    print(f"  Time: {results['total_time_sec']/60:.1f} min")
    print(f"  {'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="Sara Brain database path")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", help="Save results to JSON")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit > 0:
        questions = questions[:args.limit]

    print(f"\n  MMLU High School Biology Benchmark (tool-calling)")
    print(f"  {len(questions)} questions, model: {args.model}\n")

    if args.baseline:
        print("  --- Baseline: 3B alone, no tools ---\n")
        results = run_benchmark(questions, args.model, args.base_url, bridge=None)
        print_summary(results)
    elif args.db:
        from sara_brain.core.brain import Brain
        from sara_brain.agent.bridge import AgentBridge
        brain = Brain(args.db)
        bridge = AgentBridge(brain)
        stats = brain.stats()
        print(f"  --- Sara Brain (tool-calling) ---")
        print(f"  Brain: {args.db} ({stats['neurons']} neurons, "
              f"{stats['paths']} paths)\n")
        results = run_benchmark(questions, args.model, args.base_url,
                                bridge=bridge)
        print_summary(results)
    else:
        parser.print_help()
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump([results], f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
