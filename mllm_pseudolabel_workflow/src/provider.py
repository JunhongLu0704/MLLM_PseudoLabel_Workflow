from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResult:
    status: str
    content: str
    raw: dict[str, Any]
    latency_ms: int
    error: str = ''


class OpenAICompatProvider:
    def __init__(self, *, base_url: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

    def chat_json(self, *, system: str, user_content: Any, max_tokens: int = 512) -> ProviderResult:
        body = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            'max_tokens': int(max_tokens),
        }
        data = json.dumps(body).encode('utf-8')
        request = urllib.request.Request(
            f'{self.base_url}/chat/completions',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode('utf-8'))
            latency_ms = int((time.perf_counter() - started) * 1000)
            content = str(raw.get('choices', [{}])[0].get('message', {}).get('content', ''))
            return ProviderResult(status='ok', content=content, raw=raw, latency_ms=latency_ms)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return ProviderResult(status='provider_failed', content='', raw={}, latency_ms=latency_ms, error=f'{type(exc).__name__}: {exc}')
