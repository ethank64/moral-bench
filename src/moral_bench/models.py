"""Client for the System One API, shared by TypeSafe's Jev and a local Kev server."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

PROVIDERS = {
    "kev": {"base_url": "http://127.0.0.1:8009", "api_model": "kev-latest", "key_env": None},
    "jev": {
        "base_url": "https://api.typesafe.ai",
        "api_model": "jev-latest",
        "key_env": "TYPESAFE_API_KEY",
    },
}


@dataclass
class SystemOne:
    base_url: str
    api_model: str
    api_key: str = "local"
    timeout: float = 120.0
    max_retries: int = 5

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )

    @classmethod
    def from_provider(cls, provider: str, base_url: str | None = None) -> SystemOne:
        cfg = PROVIDERS[provider]
        key = "local"
        if cfg["key_env"]:
            key = os.environ.get(cfg["key_env"], "")
            if not key:
                raise RuntimeError(f"Set {cfg['key_env']} (see .env.example) to use {provider}.")
        return cls(base_url or cfg["base_url"], cfg["api_model"], key)

    def p_yes(self, state: str, question: str) -> float:
        """Probability that the answer to a yes/no question about `state` is yes."""
        body = {
            "state": state,
            "model": self.api_model,
            "questions": {"q": {"type": "noul", "instructions": question}},
        }
        for attempt in range(self.max_retries):
            resp = self._client.post("/v1/systemone", json=body)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return float(resp.json()["answers"]["q"]["noul"])
        resp.raise_for_status()
        raise RuntimeError("unreachable")
