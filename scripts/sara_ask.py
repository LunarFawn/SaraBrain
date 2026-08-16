#!/usr/bin/env python3
"""sara-ask: Ask Sara Brain a question, get a cited answer or honest 'I don't know.'

Usage:
    sara-ask "what causes chest pain radiating to the arm?"
    sara-ask "how deep should a well be?"
    sara-ask --brain my_medical.db "patient has high WBC and fever"
    sara-ask --teach "Aspirin inhibits platelet aggregation" --brain my_medical.db
"""
import argparse
import json
import re
import sys
import urllib.request
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sara_brain.core.brain import Brain


STOP = frozenset({'the','a','an','of','to','in','on','at','by','for','with','as','is','are',
    'was','were','be','been','do','does','did','have','has','had','will','would',
    'could','should','can','may','might','this','that','these','those','it','its',
    'they','them','their','we','our','he','she','which','who','what','when','where',
    'how','why','if','then','than','but','or','and','not','no','also','from','into',
    'through','over','under','between','about','after','before','during','following',
    'most','many','some','all','each','every','other','more','been','being','only',
    'does','make','like','much','very','patient','present'})


def content_words(text):
    return set(w.lower() for w in re.findall(r'[a-zA-Z]+', text) if w.lower() not in STOP and len(w) > 3)


def retrieve_facts(question, source_index, threshold=0.20):
    """Find relevant facts. Returns (facts, confidence)."""
    q_words = content_words(question)
    scored = []
    for words, sent in source_index:
        overlap = len(q_words & words)
        if overlap >= 2:
            scored.append((overlap, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return [], 0.0
    confidence = scored[0][0] / max(len(q_words), 1)
    facts = [sent for _, sent in scored[:5]]
    return facts, confidence


def call_cortex(question, facts, model='llama3.2:3b', base_url='http://localhost:11434'):
    """Ask the cortex to answer from facts."""
    facts_text = '\n'.join(f'  • {f}' for f in facts)
    prompt = f'''You are a knowledge assistant. Answer the question using ONLY the facts provided below.
If the facts clearly address the question, give a direct helpful answer and cite which fact supports it.
If the facts do NOT address the question, say exactly: "I don't have information about this. Please consult a specialist."

KNOWLEDGE BASE:
{facts_text}

QUESTION: {question}

Answer:'''
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': {'temperature': 0},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{base_url}/v1/chat/completions',
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            return body['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error connecting to cortex: {e}"


def load_source_index(db_path):
    """Load all source sentences from Sara's brain."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT DISTINCT source_text FROM paths WHERE source_text IS NOT NULL AND length(source_text) > 10')
    sources = [r[0] for r in c.fetchall()]
    conn.close()
    return [(content_words(s), s) for s in sources]


def teach_fact(brain_path, statement):
    """Teach Sara a new fact from a simple statement."""
    # Try to parse as "Subject relation Object" using simple patterns
    brain = Brain(brain_path)
    
    # Simple patterns
    patterns = [
        (r'^(.+?)\s+(is a|is an|is the)\s+(.+)$', 'is_a'),
        (r'^(.+?)\s+(contains|has)\s+(.+)$', 'contains'),
        (r'^(.+?)\s+(produces|generates|creates)\s+(.+)$', 'produces'),
        (r'^(.+?)\s+(requires|needs)\s+(.+)$', 'requires'),
        (r'^(.+?)\s+(causes|leads to)\s+(.+)$', 'causes'),
        (r'^(.+?)\s+(prevents|blocks|inhibits)\s+(.+)$', 'prevents'),
        (r'^(.+?)\s+(occurs in|found in|located in)\s+(.+)$', 'occurs_in'),
        (r'^(.+?)\s+(regulates|controls)\s+(.+)$', 'regulates'),
        (r'^(.+?)\s+(activates|stimulates)\s+(.+)$', 'activates'),
        (r'^(.+?)\s+(involves|includes)\s+(.+)$', 'involves'),
        (r'^(.+?)\s+(indicates|suggests|means)\s+(.+)$', 'indicates'),
        (r'^(.+?)\s+(reduces|decreases|lowers)\s+(.+)$', 'reduces'),
    ]
    
    statement_lower = statement.lower().strip().rstrip('.')
    
    for pattern, relation in patterns:
        m = re.match(pattern, statement_lower, re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            obj = m.group(3).strip()
            brain.teach_triple(subj, relation, obj, source_text=statement)
            brain.close()
            return f"✓ Learned: {subj} | {relation} | {obj}"
    
    # Fallback: store as-is with generic relation
    words = statement_lower.split()
    if len(words) >= 3:
        mid = len(words) // 2
        subj = ' '.join(words[:mid])
        obj = ' '.join(words[mid:])
        brain.teach_triple(subj, 'involves', obj, source_text=statement)
        brain.close()
        return f"✓ Learned (generic): {subj} | involves | {obj}"
    
    brain.close()
    return "✗ Could not parse. Try format: 'Subject verb Object' (e.g., 'Aspirin inhibits platelet aggregation')"


def main():
    parser = argparse.ArgumentParser(
        description="Ask Sara Brain a question or teach it new facts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sara-ask "what causes chest pain?"
  sara-ask --brain medical.db "patient WBC is 15000"
  sara-ask --teach "Aspirin inhibits platelet aggregation"
  sara-ask --teach "Well depth must reach water table" --brain wells.db
        """
    )
    parser.add_argument('question', nargs='?', help='Question to ask Sara')
    parser.add_argument('--brain', default='data/biology_hand_curated.db', help='Path to Sara brain database')
    parser.add_argument('--teach', help='Teach Sara a new fact')
    parser.add_argument('--model', default='llama3.2:3b', help='Cortex model to use')
    parser.add_argument('--url', default='http://localhost:11434', help='Ollama URL')
    parser.add_argument('--threshold', type=float, default=0.20, help='Confidence threshold for answering')
    args = parser.parse_args()

    # Teaching mode
    if args.teach:
        if not Path(args.brain).exists():
            # Create new brain
            brain = Brain(args.brain)
            brain.close()
        result = teach_fact(args.brain, args.teach)
        print(result)
        return

    # Question mode
    if not args.question:
        parser.print_help()
        return

    if not Path(args.brain).exists():
        print(f"Brain not found: {args.brain}")
        print("Create one with: sara-ask --teach 'your fact here' --brain your_brain.db")
        return

    # Load and search
    source_index = load_source_index(args.brain)
    facts, confidence = retrieve_facts(args.question, source_index, args.threshold)

    if confidence < args.threshold:
        print("\n  I don't have information about this.")
        print("  Please consult a specialist or teach me with:")
        print(f"  sara-ask --teach 'relevant fact here' --brain {args.brain}")
        return

    # Ask cortex
    answer = call_cortex(args.question, facts, model=args.model, base_url=args.url)
    
    print(f"\n  {answer}")
    print(f"\n  --- Sources ---")
    for f in facts[:3]:
        print(f"  • {f}")
    print()


if __name__ == '__main__':
    main()
