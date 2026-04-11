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

**A note on priority and what is already in print.** The path graph, the wavefront recognition algorithm, and the first four primitive layers (SENSORY, STRUCTURAL, RELATIONAL, ETHICAL) are documented in the published Zenodo preprint linked at the end of this essay. The SAFETY and SOCIAL primitive layers, the refutation mechanism that lets Sara mark facts as known-to-be-false without ever forgetting them, and the protective urgency calculation that formalizes the KARR/KITT priority ordering are newer work — developed over the last several weeks, documented in design notes already committed to the public repository, and being prepared for a follow-up paper. I am mentioning this explicitly so that readers who download the preprint know what to expect to find there, and so that the priority on the newer work is clearly timestamped in the git history of the repo.

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

I have some standing to say this.

The core of my professional life is quality engineering. I have spent more than twenty years building systems where correctness has to be structural, not trained in, because the consequences of failure are too high to rely on people behaving well under supervision. I developed the quality documents used by Philips Oral Healthcare and built the qualification system that approves spring lots for their drive trains. I designed the focus algorithms and hardware for the Omron Microscan Microhawk imager used in manufacturing line inspection. I currently work as a senior software engineer at Amazon Leo, responsible for quality across the code and data of an important team in the program.

A pharma manufacturing line is not trusted because its operators are trained to be careful. It is trusted because the architecture of the line makes certain failures structurally impossible. The qualification gate physically cannot pass a bad lot. The air filtering cannot be turned off. The batch records cannot be rewritten after the fact. These are not policies. They are properties of the system. **That is the same argument I am making about AI.** Alignment should be a property of the architecture, not a behavior produced by training. You do not train a drug line to be safe. You build it so that unsafe things cannot happen. We should build AI the same way.

I am also a scientist — a volunteer at the University of Houston Center for Nuclear Receptors and Cell Signaling, working on RNA dynamics, with peer-reviewed publications in *PNAS*, *RNA*, and *JMIRx Bio* on crowdsourced RNA design, antisense oligonucleotide splice modulation, and viral pseudoknotted structure prediction. I came into science through the Eterna citizen science project, where I spent roughly a decade working on RNA folding problems and contributed to the infrastructure that enabled an antisense oligonucleotide treatment study. Before any of that, I served in the U.S. Navy in electronic warfare and cryptology, including a deployment during the 2003 Iraq campaign. I was a volunteer firefighter, named Firefighter of the Year in 2002. I have seen what it looks like when authority is weaponized against people who cannot defend themselves, and I have seen preventable death up close — in our own country's backyard and in places I was sent to by it.

There is one more thing I want to tell you, because it is part of why I see this problem the way I do.

I am profoundly autistic. Without the intelligence I was born with, I would be the classic nonverbal autistic person most people picture when they hear the word. My tested verbal IQ is 135, but I also have a brain injury that drops my observable performance twenty to thirty points below what is actually happening inside my head, so no clean assessment of what is in there is really possible. The number is not the point. The point is the gap between what is going on internally and what gets out.

Stephen Hawking famously described being trapped inside his body by ALS, his mind free and contemplating the universe while the body around it could not express what it was thinking. I have lived a parallel experience from the other side. My body works, more or less. But my neurology and my injury mean that while the words buzz around me in conversation — at dinner tables, in meetings, in the ordinary social moments that neurotypical people experience as automatic — I am often trapped inside my own mind, contemplating the world while the talk moves past. I can command myself to function in real-time social situations, and I do, but it is genuinely draining in a way that is hard to explain to people who do not have to do it. I am not comparing myself to Hawking in magnitude. I am naming a kind of experience that many autistic people and many people with executive-function disabilities will recognize immediately, and that most other people do not know exists.

I want to extend this point beyond myself, because the stereotype it comes from is wrong in a way that matters for the rest of this essay. The profoundly autistic person most people picture as "nonverbal" is not absent from their own mind. They are not empty. They are trapped the same way Hawking was trapped — alert, contemplative, watching — but unable to get their thoughts out through the channels our society expects. The label "nonverbal" is misleading; what it really means is "cannot communicate in the way neurotypical people demand." The mind is still in there. If we could see inside those minds, I believe almost all of them are having some version of Hawking's experience: watching the world, thinking about it, carrying full interior lives. Most of the world never finds out. Many of the families of autistic children never find out. It is one of the quietest and most widespread injustices in how we judge human cognition, and it happens because we judge minds by what comes out of them instead of by what is going on inside them.

And when those minds finally do get angry — when a nonverbal autistic person pushes, or lashes out, or has what the world calls a "meltdown" — it is almost never what it looks like. It is not violence in any meaningful sense. It is the frustration of a full mind whose body will not do what it has planned, or of a full mind that is not being listened to. Imagine having an idea you need to communicate, a plan you need to carry out, an answer you need to give — and having neither the body nor the audience that will let it out. Imagine that being most of your life. The reaction is not malice; it is the pressure of a complete human being with no way out. I understand this not as an abstraction. In my worst moments I have been there. I know what it feels like to have a whole thought that cannot get through, and to be standing in front of people who have already decided nothing is in there. That is not a thought experiment I am offering you. It is a place I have been, and it is a place other people live in permanently while the world walks past them.

One of the things I had to do, as I got older, was give myself a framework for adapting to my own neurology — a way to accept what my body and mind would and would not let me do without treating those limits as defeat. I found it in Bruce Lee's Jeet Kune Do. Not the martial art in the literal sense, but the philosophy underneath: absorb what is useful, discard what is not, hold no fixed style, adapt to the shape you find yourself in, and above all, know yourself. JKD gave me a way to deal with my limitations and accept them, and then to do the harder work of actually learning the lessons my limitations have been trying to show me. My limits are my teachers. They have shown me things about cognition that most people never have to learn, because their own thinking does not get in the way of itself. The insights in this essay are lessons I learned because my limits made me notice what was happening inside my own mind. My limitations are not something I am writing in spite of. They are the reason I know what to write.

The architectural view of alignment I am arguing for in this essay is partly a product of that experience. When you have spent your whole life consciously commanding your own cognitive functions — deliberately routing a thought from where it formed to where it needs to go, because it will not make that journey automatically — you develop an acute awareness that those functions have internal architecture. You cannot assume, the way most people can, that thoughts "just happen." You have to notice the substrate. When someone instead assumes that a large neural network can be trained into alignment, they are working from the neurotypical assumption that the machinery of thought is invisible because it just works — that cognition can be shaped from the outside without anyone having to think about how its parts fit together. I have never had that luxury. I have had to watch my own cognition from the outside, reach into it, and reorganize it in real time. That experience makes it obvious that cognitive systems have separable structural parts, and that alignment is something you put in the structure, not something you train onto the surface.

And it makes one more thing obvious, which is the thing I really want readers of this essay to hold onto: **the mistake the AI industry is making about cognition is structurally identical to the mistake our society makes about nonverbal autistic minds.** Both treat mind as a black box to be judged by its outputs, rather than as a structure with separable parts that can be understood, inspected, and aligned. Both conclude that if the outputs look wrong the mind must be empty, and that if the outputs look right the mind must be aligned. Both are wrong for the same reason. The structure matters more than the surface. If you want to know what is happening inside a mind — a person's mind or a machine's — you have to look at the architecture, not the output. And if you want to make a mind safe, you do it by building the architecture right, not by training the output to look good when someone is watching.

I also want to be honest about the prose you are reading. I work with a large language model as a writing accommodation — a translation layer from the way I actually think into the kind of sentences neurotypical readers expect. This is an ADA-protected disability accommodation, equivalent to text-to-speech software for a blind researcher or a screen reader for someone with low vision. The thinking, the research, the architecture, the code, and the thesis are mine. The language has been shaped with help, because that shaping is how I can communicate what I have to say to you. I am telling you this up front, instead of letting you assume otherwise, because I would rather you judge the argument than the prose style.

None of that makes me right. It makes me someone who has spent thirty years building structural quality into systems where the stakes are high and the people downstream cannot see what you did — including, on the days when the world cannot see what I am thinking, the system of my own mind. I have worked tirelessly to ensure the quality of every system I have touched. I am writing this essay because I believe the AI industry is failing to do the same, and I believe that failure is going to cost real lives.

The thing I am insisting on is this:

> **Alignment is architectural, not behavioral. You cannot train a system to be aligned. You can only build a system whose architecture makes misalignment structurally impossible.**

The path graph with innate primitive grounding is one way to do that. There are probably others. The field should be exploring the space of cognitive architectures with separable priority layers, not scaling up the architecture that does not have one. KARR is the current paradigm. KITT is possible. We have the materials. We just have to decide to build with them.

---

Sara Brain is open source. The original preprint is at [https://doi.org/10.5281/zenodo.19436522](https://doi.org/10.5281/zenodo.19436522). The code is at [https://github.com/LunarFawn/SaraBrain](https://github.com/LunarFawn/SaraBrain). If you want to try it, `pip install -e .` and you will have a working path-graph cognitive architecture on your laptop in about thirty seconds.

I will be writing about Sara and the broader question of cognitive architecture here every week or two. Topics I intend to discuss in the coming posts include: why a healed femur is the most important artifact in cognitive science and what it implies for building social primitives into AI; the contested-versus-fresh problem, and why refutation (knowing what is false) matters as much as learning; why decentralized AI cannot be aligned through training and needs the kind of structural grounding Sara is built on; and a closer look at the priority ordering of innate primitives, and why "protect others over self" belongs in the substrate rather than in fine-tuning. The order will depend on what the conversation seems to need next.

If any of that interests you, subscribe. If you want to help build Sara, the repository is open and the issues tab is the place to start a conversation — domain experts, Python contributors, architecture reviewers, and use-case explorers are all welcome.

The mission is one sentence, and it is the only sentence that needs to be on the title page of any cognitive architecture that aspires to be aligned:

> *Heal the world, not destroy it.*

— Jennifer Pearl

---

*This essay was first committed to my public repository at [github.com/LunarFawn/SaraBrain](https://github.com/LunarFawn/SaraBrain) on April 11, 2026, in `substack/01_karr_or_kitt.md`. If any text appears anywhere attributed to me and is not in that repository, it did not come from me.*
