"""Teaching cascade — multi-layer LLM restructuring pipeline.

Takes arbitrary source prose and produces clean subject-verb-object
claims Sara's parser accepts. Architecture: multiple 1B rod-and-cone
sensors feeding per-type 3B integrators feeding a final 3B
synthesizer. The LLM never teaches — it only restructures content
already present in the source.
"""
