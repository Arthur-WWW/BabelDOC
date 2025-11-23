"""DeepBridge adapter package.

This exposes the orchestrator used by the CLI entry point (`main.py`).
"""

from .pipeline import run_hybrid_pipeline

__all__ = ["run_hybrid_pipeline"]
