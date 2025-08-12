from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional


class BaseModelClient:
    name: str = "base"

    async def generate(self, prompt: str, **kw) -> Dict[str, Any]:
        raise NotImplementedError


class LocalEcho(BaseModelClient):
    name = "local-echo"

    async def generate(self, prompt: str, **kw):
        start = time.time()
        time.sleep(0.01)
        out = prompt[:200]  # naive echo for offline testing
        return {"output": out, "latency_s": time.time() - start, "cost_usd": 0.0}


class OpenAIClient(BaseModelClient):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    async def generate(self, prompt: str, **kw):
        # Placeholder: sketch structure without making real calls.
        # Swap with openai SDK usage where allowed.
        return {
            "output": f"[{self.model}] {prompt[:200]}",
            "latency_s": 0.0,
            "cost_usd": 0.0,
        }
