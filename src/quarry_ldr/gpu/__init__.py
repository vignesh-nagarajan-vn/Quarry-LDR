"""GPU residency and local model access.

Everything that touches the GPU goes through :class:`quarry_ldr.gpu.arbiter.VramArbiter`.
No module outside this package may call ``.to("cuda")`` or load a model directly.
"""
