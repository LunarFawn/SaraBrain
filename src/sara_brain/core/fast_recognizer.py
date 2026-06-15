import ctypes
import os
import sys
from typing import List, Dict, Optional
from ..models.neuron import Neuron
from .recognizer import Recognizer, _NON_PROPAGATING_RELATIONS

class ResultNode(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("weight", ctypes.c_float)]

class FastRecognizer(Recognizer):
    """High-performance C++ backend for wavefront propagation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._engine = None
        self._lib = None
        self._load_lib()
        self._init_engine()

    def _load_lib(self):
        lib_path = os.path.join(os.path.dirname(__file__), "sara_engine.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"C++ engine library not found at {lib_path}. Run 'make' first.")
        self._lib = ctypes.CDLL(lib_path)
        
        # Setup argument and return types
        self._lib.engine_create.restype = ctypes.c_void_p
        self._lib.engine_destroy.argtypes = [ctypes.c_void_p]
        self._lib.engine_add_segment.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self._lib.engine_clear.argtypes = [ctypes.c_void_p]
        self._lib.engine_propagate.argtypes = [
            ctypes.c_void_p, # engine
            ctypes.c_int,    # start_node
            ctypes.c_int,    # max_depth
            ctypes.c_float,  # min_strength
            ctypes.c_bool,   # bidirectional
            ctypes.POINTER(ResultNode), # out_results
            ctypes.c_int     # max_results
        ]
        self._lib.engine_propagate.restype = ctypes.c_int

    def _init_engine(self):
        if self._engine:
            self._lib.engine_destroy(self._engine)
        self._engine = self._lib.engine_create()
        
        # Load segments from repo into C++ engine
        # This is a one-time cost per Recognizer instance
        print(f"[FastRecognizer] Loading graph into C++ engine...", file=sys.stderr)
        segments = self.segment_repo.list_all()
        for seg in segments:
            is_prop = seg.relation not in _NON_PROPAGATING_RELATIONS
            self._lib.engine_add_segment(self._engine, seg.source_id, seg.target_id, seg.strength, is_prop)
        print(f"[FastRecognizer] Loaded {len(segments)} segments.", file=sys.stderr)

    def __del__(self):
        if self._engine and self._lib:
            self._lib.engine_destroy(self._engine)

    def propagate_into(self, seed_labels: list[str], short_term,
                       min_strength: float | None = None,
                       exact_only: bool = True) -> None:
        """Optimized version using C++ engine."""
        seeds = []
        for label in seed_labels:
            n = self.neuron_repo.resolve(label.strip().lower(), exact_only=exact_only)
            if n is not None:
                seeds.append(n)
        if not seeds:
            return

        effective_min = self.min_strength if min_strength is None else min_strength
        
        # Max results buffer
        max_res = 50000 
        result_buffer = (ResultNode * max_res)()

        for seed in seeds:
            count = self._lib.engine_propagate(
                self._engine, seed.id, self.max_depth, effective_min,
                False, # propagate_into is typically forward-only in current Python logic
                result_buffer, max_res
            )
            for i in range(count):
                res = result_buffer[i]
                if res.id == seed.id: continue
                short_term.add_convergence(res.id, res.weight, seed.id)

    def _propagate(self, start: Neuron,
                   min_strength: float | None = None,
                   bidirectional: bool = False) -> dict[int, list[list[Neuron]]]:
        """Override to use C++ engine when possible.
        
        NOTE: The C++ engine currently only returns best weights, not full paths.
        For functions that NEED paths (like recognize or trace), this will 
        fall back to the slow Python BFS.
        """
        # For now, we only use C++ for the best-weight paths like propagate_into.
        # If we need paths, we use the super class.
        # However, we can trick it by returning a mock path if we only care about the weight.
        # But let's stay safe for now.
        return super()._propagate(start, min_strength, bidirectional)

    # propagate_echo uses self._propagate. To optimize it, we should override it too.
    def propagate_echo(self, seed_labels: list[str], short_term,
                       max_rounds: int = 2,
                       min_strength: float | None = None,
                       exact_only: bool = True,
                       top_k: int = 10) -> None:
        """Optimized echo using C++ engine."""
        effective_min = self.min_strength if min_strength is None else min_strength

        used_ids = set()
        current_seeds = []
        for label in seed_labels:
            n = self.neuron_repo.resolve(label.strip().lower(), exact_only=exact_only)
            if n is not None and n.id not in used_ids:
                current_seeds.append(n)
                used_ids.add(n.id)
        if not current_seeds: return

        max_res = 50000
        result_buffer = (ResultNode * max_res)()

        for _round in range(max_rounds):
            for seed in current_seeds:
                count = self._lib.engine_propagate(
                    self._engine, seed.id, self.max_depth, effective_min,
                    True, # Echo is bidirectional
                    result_buffer, max_res
                )
                for i in range(count):
                    res = result_buffer[i]
                    if res.id == seed.id: continue
                    short_term.add_convergence(res.id, res.weight, seed.id)

            intersections = short_term.intersections(min_sources=1)
            next_candidates = []
            for nid, weight, _count in intersections:
                if nid not in used_ids:
                    # Hub Inhibition
                    out_count = len(self.segment_repo.get_outgoing(nid))
                    in_count = len(self.segment_repo.get_incoming(nid))
                    connectivity = out_count + in_count
                    inhibited_weight = weight / (connectivity + 1)
                    
                    # We still need the Neuron object for the next round
                    n = self.neuron_repo.get_by_id(nid)
                    if n:
                        next_candidates.append((n, inhibited_weight))
            
            if not next_candidates: break
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            current_seeds = [c[0] for c in next_candidates[:top_k]]
            for n in current_seeds: used_ids.add(n.id)
