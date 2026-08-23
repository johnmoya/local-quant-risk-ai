"""Thin HTTP client for a local Ollama instance (default model: Qwen3 8B).
No risk logic here — request/response plumbing and error handling only
(timeouts, connection failures must surface as typed exceptions so
api/routers/explain.py can fail independently of the numeric endpoints).

Implemented in M7.
"""
