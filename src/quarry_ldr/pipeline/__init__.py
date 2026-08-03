"""Pipeline stages and the orchestrating state machine.

Stage functions are pure-ish async functions over injected components; the
orchestrator in run.py owns sequencing, checkpointing, and the iteration loop.
"""
