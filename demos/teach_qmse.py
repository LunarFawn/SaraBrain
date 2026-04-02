# -*- coding: utf-8 -*-
"""Teach Sara the QMSE document into persistent storage."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sara_brain.config import default_db_path
from sara_brain.core.brain import Brain

db = default_db_path()
print(f"Using: {db}")

b = Brain(db)

facts = [
    "quality manufacturing software engineering is a coding philosophy",
    "QMSE is quality manufacturing software engineering",
    "coding for quality is quality manufacturing software engineering",
    "QMSE is inspired by martial arts philosophy",
    "formless form is the goal of QMSE",
    "kung fu means skill acquired through hard work",
    "object oriented programming is a frontend pattern",
    "functional oriented programming is a backend pattern",
    "frontend is user facing code",
    "backend is heavy lifting code",
    "coupling is interconnectedness in code",
    "low coupling is a goal",
    "hardcoding is never acceptable",
    "obfuscation through parameterization is acceptable",
    "full customization at runtime is the goal",
    "comments are important",
    "no comments is bad advice",
    "short variable names are bad practice",
    "single letter variables are bad practice",
    "long descriptive names are preferred",
    "character_index is better than i",
    "PEP8 line limit is 70 characters",
    "PEP8 line limit is too small",
    "list comprehension is faster than loops",
    "style guides are references not bibles",
    "nested if statements are useful",
    "hatred of nested ifs is unfounded",
    "QMSE is for manufacturing environments",
    "QMSE is for data science environments",
    "QMSE is not for web coding",
    "the author is peer review published in computational biology",
    "the author is an autistic transwoman",
]

taught = 0
skipped = 0
for fact in facts:
    result = b.teach(fact)
    if result:
        taught += 1
        print(f"  + {fact}")
    else:
        skipped += 1
        print(f"  - (unparseable) {fact}")

print(f"\nTaught: {taught}, Skipped: {skipped}")

stats = b.stats()
print(f"Brain: {stats['neurons']} neurons, {stats['segments']} segments, {stats['paths']} paths")

b.close()
print("Sara is asleep. Knowledge persisted.")
