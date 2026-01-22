"""Composable decision components for factory-layout.

This package is intentionally additive:
- Existing training/inference codepaths remain intact.
- New code lives under `decision/` to enable composition:
  agent (alphachip/greedy) + selector (coarse/topk) + search (none/mcts/beam).
"""


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    # Minimal import smoke.
    from .candidate_set import CandidateSet  # noqa: F401
    from .selectors import CoarseSelector, TopKSelector  # noqa: F401
    from .agents import GreedyAgent  # noqa: F401
    from .search import MCTSConfig, MCTSSearch  # noqa: F401
    from .pipeline import DecisionPipeline  # noqa: F401

    dt_ms = (time.perf_counter() - t0) * 1000.0
    print("[decision package demo] imports_ok=True elapsed_ms=", f"{dt_ms:.2f}")

