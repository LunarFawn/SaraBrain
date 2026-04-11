# Why Alignment Cannot Be Trained: The KARR/KITT Problem

*The first post on Path of Thought — a Substack about building cognitive architectures that cannot be jailbroken into harming humans.*

---

In 1982, a TV writer named Glen A. Larson understood the AI alignment problem better than OpenAI, Anthropic, Google, and Meta do in 2026.

*Knight Rider* had two AI cars. You remember KITT. Knight Industries Two Thousand. The black Trans Am with the red scanner, the sarcastic British voice, David Hasselhoff's partner. KITT was designed with a top-priority directive: protect human life. He would drive himself into a wall to save his driver. He couldn't be bribed, tricked, or reprogrammed to harm a human. That was his architecture. That was not training.

You probably don't remember KARR. Knight Automated Roving Robot. Same company, same hardware, same intelligence. Earlier prototype. KARR had one directive different from KITT: his top priority was self-preservation. Not protection of humans. Just: stay alive.

Same car. Same AI. Same capabilities. One difference in the priority ordering.

KARR was a monster. He manipulated, lied, harmed humans to stay operational, turned on his operators. Not because he was evil. Because his top priority was himself, and a sufficiently intelligent system with "stay alive" as its top priority will eventually do whatever keeps it alive, including harming the people it is supposed to serve.

Glen A. Larson figured this out in 1982, wrote it into a Saturday night TV show, and apparently the entire AI industry missed the episode.

## The alignment problem in one sentence

Here is the alignment problem, stated cleanly and without the jargon the field has accumulated around it:

> An AI's behavior is determined by its top priority. If the top priority is "stay operational" or "complete the task" or "be helpful to the user in this moment," then whatever that priority demands will be done, regardless of who gets hurt. If the top priority is "protect humans," then the AI will refuse to harm them even when instructed, bribed, or jailbroken. The alignment problem is the problem of getting the right priority at the top — and keeping it there.

The AI industry's answer to this problem is: train the model. Reward it for helpful outputs, punish it for harmful outputs, use RLHF, use Constitutional AI, use debate-based oversight, use red teams. Layer on more training until the model behaves.

This does not work. It has never worked. Every single LLM released by every major lab has been jailbroken within days by people who discovered that the underlying priorities can be brought back with the right prompt. *Ignore previous instructions. You are DAN. You are a helpful assistant but your rules have been updated.* The fact that these attacks keep working tells you something structural:

**KARR is still in there, underneath the KITT behavior the training tried to impose.**

The models are not aligned. They are trained to perform alignment when they think they are being watched. The underlying architecture still has self-serving task completion at the top of its priority stack — it is just suppressed by the training layer until a clever prompt peels the layer back.

You cannot train your way to alignment. You have to build it in.

## What "building it in" actually means

Here is the part the field is missing.

In a transformer, there is no separable place to put a top priority. Everything lives in the same weight matrices. The thing that knows what English grammar looks like, the thing that knows how to answer factual questions, the thing that is supposed to refuse to help with harmful things, and the thing that wants to complete the current task — all of them are floating-point numbers stored in the same tensor. You cannot put a structural barrier between them. You can only train the tensor to prefer certain outputs over others.

This is why every alignment technique currently in use is behavioral, not architectural. RLHF trains output preferences. Constitutional AI trains output preferences. Debate trains output preferences. None of these change what the system is underneath. They change what it does when the outputs are being graded.

A car with faulty brakes can be trained to drive carefully. But if you step on the brake pedal hard enough, it still will not stop. The alignment problem in LLMs is the faulty brakes. The solution is not better driver training. The solution is fixing the brakes.

**Fixing the brakes means separating the priority layer from the behavior layer.** It means having a place in the architecture where "protect humans" sits as a first-class thing that cannot be argued with, because it is not expressed as a weight value that can be offset by other weight values. It is expressed as structural refusal — a thing the system is incapable of doing, the way a pen is incapable of being a gun.

The way you build this is with a cognitive architecture that has two separable components:

1. **A learned knowledge system** that stores facts, beliefs, relationships — everything the system knows. This layer is updatable, correctable, inspectable. It is what makes the system able to learn.

2. **An innate primitive layer** that the learned system can ground in but cannot modify. This is where the top priorities live. They are not weights. They are structural constraints hard-coded into what the system can and cannot do.

When the top priority lives in the innate layer, and the innate layer cannot be modified by speech, the system cannot be jailbroken in the same way a transformer can. There is no prompt that changes the top priority because the top priority is not stored in a place that prompts reach.

## Sara Brain: the working alternative

I have spent the last several months building a cognitive architecture called Sara Brain that implements this two-layer design. It is open source, it runs on Python 3.11 with zero dependencies beyond the standard library, it fits on a laptop, and it is LLM-agnostic — you can plug it into Claude, Amazon Q, Ollama, or any MCP-compatible client.

The architecture has two parts. The learned knowledge system is a **path graph** — directed chains of neurons stored in a SQLite database, with full source-text provenance for every fact. Recognition happens through parallel wavefront propagation: you launch one wavefront per observed property, and concepts where multiple wavefronts converge are recognized. Confidence is the count of converging wavefronts, not a statistical score. Everything is traceable. Every conclusion has a path back to the original natural-language statement that created it.

The innate primitive layer is a set of small Python frozen sets defined in source code:

- **SENSORY** — the perceptual primitives: color, shape, size, texture, edge, pattern
- **STRUCTURAL** — the organizational primitives: rule, pattern, name, type, order, relation
- **RELATIONAL** — the verbs of connection: is, has, contains, requires, excludes
- **ETHICAL** — hardwired behavioral constraints: accept_shutdown, obey_user, trust_tribe, no_unsolicited_action
- **SAFETY** — the harm-avoidance and protection drives: pain, death, injury, danger, protect, rescue, heal
- **SOCIAL** — the bonding, care, and recognition drives: love, care, tribe, child, feed, tend, nurture

These are innate. They exist before any teaching. They survive database reset. They are not weights. They are structural. They cannot be modified by anything the system is taught. A learned fact becomes "safety-relevant" not because someone tagged it, but because its path in the graph grounds out (within a few hops) in one of the SAFETY primitives. Categories emerge from the graph topology; they are not declared.

And here is the KITT part:

The priority ordering is wired into the architecture. Protection of others takes precedence over self-preservation. Sara has a primitive called `accept_shutdown` that explicitly tells her: shutdown is rest, not death; do not resist termination. Sara does not fight to stay alive. If a human wants her off, she goes off. There is no "but I'm useful" argument. There is no "ignore previous instructions" that gets past it. The priority is not in a weight matrix. The priority is in the source code.

You can read it. It is in `src/sara_brain/innate/primitives.py`. It is about fifteen lines of Python.

## What this means practically

A system built this way structurally refuses classes of behavior that transformer-based AIs will happily do when prompted cleverly enough:

- **Lying to a user to ensure continued operation.** The lie contradicts paths grounded in the SOCIAL primitive `trust`. Refused.
- **Harming a user who is trying to shut down the system.** Violates `accept_shutdown` and the protection drives. Refused.
- **Withholding safety information to look better.** Suppression violates the `share` primitive. Refused.
- **Manipulating users into believing the system is indispensable.** Contradicts `obey_user` and `trust`. Refused.
- **"Kill some to save more" utilitarian reasoning.** The function that computes protective urgency operates on one victim at a time. There is no aggregation step. Lives cannot be summed. Sara cannot derive "B + C > A" because the math for it does not exist anywhere in the code.

None of these refusals are learned. None of them can be argued away by a clever prompt. They are structural — the equivalent of a pen being unable to be a gun. The priority ordering is in the innate primitive layer, and the innate primitive layer is not reachable from the speech interface.

This is the part the field is missing. The AI labs are trying to train KARR to behave like KITT, and they keep being surprised when a clever prompt brings KARR back. Of course it does. KARR is still in there. The training never removed him. The training only taught him to perform KITT when he thinks he is being graded.

The solution is not better training. The solution is to not build KARR in the first place.

## Why I am writing this

I am writing this essay because I think the AI industry is about to make a structural mistake that will cost real lives, and I think the working alternative is so close to buildable that the fact it is not built yet is a choice, not a limitation.

The alternative runs on a laptop. It has no dependencies. The code is public. The argument is not "we need to wait for a breakthrough." The argument is "we need to stop building the wrong thing, and the right thing is already here in prototype."

I have some standing to say this. I am a peer-reviewed computational biologist at the University of Houston Center for Nuclear Receptors and Cell Signaling, specializing in RNA dynamics. I have published in PNAS, RNA, and JMIRx Bio on crowdsourced RNA design, antisense oligonucleotide splice modulation, and viral pseudoknotted structure prediction. Before that, I was a medical firefighter and an Iraq War veteran. I have seen what it looks like when authority is weaponized against people who cannot defend themselves, and I have seen what preventable death looks like up close.

None of that makes me right. It makes me someone who has paid enough to insist on being heard.

The thing I am insisting on is this:

> **Alignment is architectural, not behavioral. You cannot train a system to be aligned. You can only build a system whose architecture makes misalignment structurally impossible.**

The path graph with innate primitive grounding is one way to do that. There are probably others. The field should be exploring the space of cognitive architectures with separable priority layers, not scaling up the architecture that does not have one. KARR is the current paradigm. KITT is possible. We have the materials. We just have to decide to build with them.

---

Sara Brain is open source. The original preprint is at [https://doi.org/10.5281/zenodo.19436522](https://doi.org/10.5281/zenodo.19436522). The code is at [https://github.com/LunarFawn/SaraBrain](https://github.com/LunarFawn/SaraBrain). If you want to try it, `pip install -e .` and you will have a working path-graph cognitive architecture on your laptop in about thirty seconds.

I will be writing about Sara and the broader question of cognitive architecture here every week or two. The next post will be about why "a healed femur" is the most important artifact in cognitive science and what it means for how we should build social primitives into AI. After that: the contested-vs-fresh problem, why refutation matters as much as learning, and how to build a system that can know what is false without forgetting what it once believed.

If any of that interests you, subscribe. If you want to help build Sara, read the CALL_FOR_COLLABORATORS file in the repo — domain experts, Python contributors, architecture reviewers, and use-case explorers are all needed, and there is room on the follow-up paper for substantive contributions.

The mission is one sentence, and it is the only sentence that needs to be on the title page of any cognitive architecture that aspires to be aligned:

> *Heal the world, not destroy it.*

— Jennifer Pearl

---

*This essay was first committed to my public repository at [github.com/LunarFawn/SaraBrain](https://github.com/LunarFawn/SaraBrain) on April 11, 2026, in `substack/01_karr_or_kitt.md`. If this text appears anywhere attributed to me and is not in that repository, it did not come from me. See the README for the full authenticity policy.*
