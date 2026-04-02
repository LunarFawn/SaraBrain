# -*- coding: utf-8 -*-
"""Dump everything Sara knows after QMSE ingestion."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from collections import defaultdict
from sara_brain.core.brain import Brain

b = Brain()

facts = [
    "quality manufacturing software engineering is a coding philosophy",
    "QMSE is quality manufacturing software engineering",
    "coding for quality is quality manufacturing software engineering",
    "QMSE uses object oriented wrappers around functional backends",
    "QMSE is inspired by martial arts philosophy",
    "kung fu means skill acquired through hard work",
    "formless form is the goal of QMSE",
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
    "coupling is interconnectedness in code",
    "global variables cause coupling",
    "imports cause coupling",
    "low coupling is a goal",
    "functional backend reduces coupling",
    "hardcoding is never acceptable",
    "values should be assigned to variables",
    "default values belong in function signatures",
    "obfuscation through parameterization is acceptable",
    "enums should replace strings in CLI calls",
    "enums reduce complexity for users",
    "raise_exception should be a parameter not hardcoded",
    "never hardcode exceptions",
    "full customization at runtime is the goal",
    "comments are important",
    "code should be self documenting",
    "comments should summarize intent",
    "comments before new code blocks describe goals",
    "comments act as a notebook for intentions",
    "no comments is bad advice",
    "variable names should match human language",
    "short variable names are bad practice",
    "single letter variables are bad practice",
    "long descriptive names are preferred",
    "character_index is better than i",
    "PEP8 line limit is 70 characters",
    "PEP8 line limit is too small",
    "PEP8 line limit prevents list comprehension",
    "list comprehension is faster than loops",
    "PEP8 line limit compensates for small monitors",
    "code should not compensate for bad hardware",
    "style guides are references not bibles",
    "style guides should not override situational judgment",
    "nested if statements are useful",
    "hatred of nested ifs is unfounded",
    "QMSE is for manufacturing environments",
    "QMSE is for data science environments",
    "QMSE is not for web coding",
    "web coding follows rote processes",
    "QMSE follows ISO9000 principles",
    "QMSE applies to FDA class 1 medical devices",
    "QMSE applies to FAA regulated systems",
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
for f in facts:
    b.teach(f)

# All neurons by type
neurons = b.neuron_repo.list_all()
print(f"=== Sara knows {len(neurons)} neurons ===\n")

by_type = defaultdict(list)
for n in neurons:
    by_type[n.neuron_type.value].append(n.label)

for ntype in ["concept", "property", "relation"]:
    labels = sorted(by_type.get(ntype, []))
    print(f"--- {ntype.upper()} ({len(labels)}) ---")
    for label in labels:
        print(f"  {label}")
    print()

# All paths
paths = b.path_repo.list_all()
print(f"=== {len(paths)} learned paths ===\n")
for p in paths:
    steps = b.path_repo.get_steps(p.id)
    chain = []
    for step in steps:
        seg = b.segment_repo.get_by_id(step.segment_id)
        if seg:
            if not chain:
                src = b.neuron_repo.get_by_id(seg.source_id)
                if src:
                    chain.append(src.label)
            tgt = b.neuron_repo.get_by_id(seg.target_id)
            if tgt:
                chain.append(tgt.label)
    arrow = " -> ".join(chain)
    print(f"  {arrow}")
    print(f"    \"{p.source_text}\"")

b.close()
