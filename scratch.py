from sara_brain.cortex.transformer.chat import parse_edges_from_gathered
from sara_brain.cortex.transformer.synth_data import build_slot_mapping, SynthExample
from sara_brain.cortex.transformer.inference_synth import _expand_slots
edges = parse_edges_from_gathered([{"call":{}, "result": "'gymnosperm' --[interacts_with]--> 'seed plants include_attribute'"}])
ex = SynthExample(edges=edges, prose="", subject="")
mapping = build_slot_mapping(ex)
print(mapping)
print(_expand_slots(["<C0>", "interacts", "with", "<C1>"], mapping))
