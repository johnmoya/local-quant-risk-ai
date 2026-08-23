"""FastAPI service layer: thin HTTP adapters over the risk engine and the
LLM explainer. No math and no LLM orchestration logic lives here — routers
parse requests, call quant_risk_ai.risk / quant_risk_ai.llm, and serialize
responses.
"""
