import torch
from sara_brain.cortex.transformer.chat import parse_edges_from_gathered
from sara_brain.cortex.transformer.inference_synth import synthesize_cluster, build_slot_mapping, SynthExample, load_synth_checkpoint
gathered = [{'call': {'tool': 'brain_explore', 'args': {'label': 'gymnosperm', 'depth': 1}}, 'result': """'gymnosperm' --[part_of]--> 'gymnosperm ancestor'
'gymnosperm' --[part_of]--> 'gymnosperm pollination'
'gymnosperm' --[interacts_with]--> 'seed plants include_attribute'
'gymnosperm' --[is_a]--> 'ginkgoale_attribute'
'gymnosperm' --[is_a]--> 'sporophyte_attribute'
'gymnosperm' --[causes]--> 'ginkgo biloba_attribute'
'gymnosperm' --[involves]--> 'life_attribute'
'gymnosperm' --[transforms_into]--> 'divided_attribute'
"""}]
edges = parse_edges_from_gathered(gathered)
print(f"Number of edges: {len(edges)}")
ex = SynthExample(edges=edges, prose="", subject="")
mapping = build_slot_mapping(ex)
print(f"Mapping: {mapping}")
model = load_synth_checkpoint("src/sara_brain/cortex/checkpoints/hamroby_sum_biology_synth_pairs_002000.pt", torch.device("cpu"))
prose = synthesize_cluster(model, edges, torch.device("cpu"))
print(f"Prose: {prose}")
