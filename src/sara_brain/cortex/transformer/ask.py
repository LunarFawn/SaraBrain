"""End-to-end: ask Sara a question through the cortex router.

  question -> CortexRouter -> tool call -> brain.db -> formatted result

No LLM in the loop. The grammar LM + classifier head pick the tool;
rule extractor pulls args; the substrate executor runs the call.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sara_brain.core.brain import Brain
from sara_reader.tools import execute_tool

from .router import CortexRouter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--question", "-q", type=str, required=True)
    p.add_argument("--brain", type=Path, required=True)
    p.add_argument("--grammar-ckpt", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/grammar_base_015000.pt"))
    p.add_argument("--head-ckpt", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/router_head.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    router = CortexRouter(
        grammar_ckpt=args.grammar_ckpt,
        head_ckpt=args.head_ckpt,
        substrate_db=args.brain,
        device=args.device,
    )
    brain = Brain(str(args.brain))

    decision = router.route(args.question)
    if not args.quiet:
        print(f"Q:    {args.question}")
        print(f"tool: {decision.tool}  (cls_conf={decision.classifier_confidence:.2f})")
        print(f"args: {decision.args}")
        print(f"why:  {decision.rationale}")
        print()

    result = execute_tool(brain, decision.tool, decision.args)
    print(result)


if __name__ == "__main__":
    main()
