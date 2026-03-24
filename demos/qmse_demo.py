# -*- coding: utf-8 -*-
"""Teach Sara the QMSE document and take her for a test drive."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from sara_brain.core.brain import Brain

b = Brain()

facts = [
    # Core philosophy
    "quality manufacturing software engineering is a coding philosophy",
    "QMSE is quality manufacturing software engineering",
    "coding for quality is quality manufacturing software engineering",
    "QMSE uses object oriented wrappers around functional backends",
    "QMSE is inspired by martial arts philosophy",
    "kung fu means skill acquired through hard work",
    "formless form is the goal of QMSE",

    # Architecture pattern
    "object oriented programming is a frontend pattern",
    "functional oriented programming is a backend pattern",
    "frontend is user facing code",
    "backend is heavy lifting code",
    "frontend calls backend",
    "backend returns values",
    "backend functions do not share global variables",
    "hybrid OOP-FOP reduces coupling",
    "hybrid OOP-FOP increases extensibility",
    "hybrid OOP-FOP increases scalability",

    # Coupling
    "coupling is interconnectedness in code",
    "global variables cause coupling",
    "imports cause coupling",
    "low coupling is a goal",
    "functional backend reduces coupling",

    # No hardcoding
    "hardcoding is never acceptable",
    "values should be assigned to variables",
    "default values belong in function signatures",
    "obfuscation through parameterization is acceptable",
    "enums should replace strings in CLI calls",
    "enums reduce complexity for users",
    "raise_exception should be a parameter not hardcoded",
    "never hardcode exceptions",
    "full customization at runtime is the goal",

    # Comments and documentation
    "comments are important",
    "code should be self documenting",
    "comments should summarize intent",
    "comments before new code blocks describe goals",
    "comments act as a notebook for intentions",
    "no comments is bad advice",

    # Naming
    "variable names should match human language",
    "short variable names are bad practice",
    "single letter variables are bad practice",
    "long descriptive names are preferred",
    "character_index is better than i",

    # PEP8 criticism
    "PEP8 line limit is 70 characters",
    "PEP8 line limit is too small",
    "PEP8 line limit prevents list comprehension",
    "list comprehension is faster than loops",
    "PEP8 line limit compensates for small monitors",
    "code should not compensate for bad hardware",
    "style guides are references not bibles",
    "style guides should not override situational judgment",

    # Nested IFs
    "nested if statements are useful",
    "hatred of nested ifs is unfounded",

    # Manufacturing vs web
    "QMSE is for manufacturing environments",
    "QMSE is for data science environments",
    "QMSE is not for web coding",
    "web coding follows rote processes",
    "QMSE follows ISO9000 principles",
    "QMSE applies to FDA class 1 medical devices",
    "QMSE applies to FAA regulated systems",

    # Author background
    "the author has RNA research experience",
    "the author has Stanford association",
    "the author has University of Houston association",
    "the author is peer review published in computational biology",
    "the author developed machine vision focus algorithms",
    "the author has 15 years coding experience",
    "the author has ADHD and dyslexia",
    "the author is an autistic transwoman",
    "the author served in US Navy electronic warfare",
]

taught = 0
for fact in facts:
    result = b.teach(fact)
    if result:
        taught += 1

print(f"Taught Sara {taught} facts from the QMSE document.")
print()

stats = b.stats()
print(f"Brain stats: {stats['neurons']} neurons, {stats['segments']} segments, {stats['paths']} paths")
print()

# Recognition tests
print("--- Recognition Tests ---")
for inputs in [
    "manufacturing, data science, ISO9000",
    "functional, backend, no global variables",
    "hardcoding, never",
    "comments, important, intent",
    "long, descriptive, human language",
    "too small, prevents list comprehension",
    "object oriented, functional, hybrid",
    "coding philosophy, martial arts",
    "bad practice, short",
]:
    results = b.recognize(inputs)
    if results:
        top = results[0]
        print(f"  recognize [{inputs}] -> {top.neuron.label} (paths: {top.converging_paths})")
    else:
        print(f"  recognize [{inputs}] -> (nothing)")

print()

# Trace QMSE
print("--- Trace: QMSE ---")
traces = b.trace("quality manufacturing software engineering")
for t in traces[:8]:
    chain = " -> ".join(n.label for n in t.neurons)
    print(f"  {chain}")

print()

# Why hardcoding?
print("--- Why: hardcoding ---")
whys = b.why("hardcoding")
for w in whys:
    print(f"  [{w.source_text}]")

print()

# Why QMSE?
print("--- Why: QMSE ---")
whys = b.why("quality manufacturing software engineering")
for w in whys:
    print(f"  [{w.source_text}]")

print()

# Similar to QMSE
print("--- Similar to: QMSE ---")
sims = b.get_similar("quality manufacturing software engineering")
for s in sims:
    print(f"  {s.source_label} <-> {s.target_label} (shared: {s.shared_targets})")

b.close()
