"""Command implementations for the REPL."""

from __future__ import annotations

from ..core.brain import Brain
from . import formatters


def cmd_teach(brain: Brain, args: str) -> str:
    if not args.strip():
        return "  Usage: teach <statement>"
    result = brain.teach(args)
    if result is None:
        return "  Could not parse statement. Try: teach apples are red"
    return formatters.format_learn_result(
        result.path_label, result.segments_created, result.neurons_created
    )


def cmd_recognize(brain: Brain, args: str) -> str:
    if not args.strip():
        return "  Usage: recognize <input1>, <input2>, ..."
    results = brain.recognize(args)
    return formatters.format_recognition(results)


def cmd_why(brain: Brain, args: str) -> str:
    if not args.strip():
        return "  Usage: why <concept>"
    traces = brain.why(args.strip())
    return formatters.format_why(args.strip(), traces)


def cmd_trace(brain: Brain, args: str) -> str:
    if not args.strip():
        return "  Usage: trace <neuron>"
    traces = brain.trace(args.strip())
    return formatters.format_trace(args.strip(), traces)


def cmd_neurons(brain: Brain, _args: str) -> str:
    neurons = brain.neuron_repo.list_all()
    return formatters.format_neurons(neurons)


def cmd_paths(brain: Brain, _args: str) -> str:
    paths = brain.path_repo.list_all()
    return formatters.format_paths(paths)


def cmd_stats(brain: Brain, _args: str) -> str:
    stats = brain.stats()
    return formatters.format_stats(stats)


def cmd_similar(brain: Brain, args: str) -> str:
    if not args.strip():
        return "  Usage: similar <neuron>"
    links = brain.get_similar(args.strip())
    return formatters.format_similarities(links)


def cmd_analyze(brain: Brain, _args: str) -> str:
    links = brain.analyze_similarity()
    if not links:
        return "  No new similarities found."
    return f"  Found {len(links)} similarity link(s):\n" + formatters.format_similarities(links)


def cmd_define(brain: Brain, args: str) -> str:
    parts = args.strip().split()
    if len(parts) < 2:
        return "  Usage: define <association> <question_word>\n  Example: define taste how"
    name, question_word = parts[0], parts[1]
    neuron = brain.define_association(name, question_word)
    return formatters.format_define_result(neuron.label, question_word)


def cmd_describe(brain: Brain, args: str) -> str:
    # Parse: "<association> as <prop1>, <prop2>, ..."
    parts = args.split(" as ", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return "  Usage: describe <association> as <prop1>, <prop2>, ..."
    name = parts[0].strip()
    properties = [p.strip() for p in parts[1].split(",") if p.strip()]
    if not properties:
        return "  Usage: describe <association> as <prop1>, <prop2>, ..."
    try:
        registered = brain.describe_association(name, properties)
    except ValueError as e:
        return f"  {e}"
    return formatters.format_describe_result(name, registered)


def cmd_associations(brain: Brain, _args: str) -> str:
    assocs = brain.list_associations()
    return formatters.format_associations(assocs)


def cmd_query(brain: Brain, question_word: str, args: str) -> str:
    """Handle: <question_word> <subject> <association>"""
    parts = args.strip().split()
    if len(parts) < 2:
        return f"  Usage: {question_word} <concept> <association>"
    subject, association = parts[0], parts[1]
    results = brain.query_association(subject, association)
    return formatters.format_query(question_word, subject, association, results)


def cmd_questions(brain: Brain, _args: str) -> str:
    question_words = brain.list_question_words()
    return formatters.format_questions(question_words)


def cmd_categorize(brain: Brain, args: str) -> str:
    parts = args.strip().split()
    if len(parts) < 2:
        return "  Usage: categorize <concept> <category>\n  Example: categorize apple item"
    label, category = parts[0], parts[1]
    brain.categorize(label, category)
    return formatters.format_categorize(label, category)


def cmd_categories(brain: Brain, _args: str) -> str:
    categories = brain.list_categories()
    return formatters.format_categories(categories)
