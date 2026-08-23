"""Typed exceptions shared across layers (e.g. InsufficientDataError,
SingularCovarianceError, OllamaUnavailableError, NumericConsistencyError),
so the API layer can map them to the right HTTP status instead of leaking
raw numpy/httpx errors.

Populated incrementally starting M2, as each layer's edge cases are
implemented.
"""
