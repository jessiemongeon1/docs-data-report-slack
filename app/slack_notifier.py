from __future__ import annotations

from typing import Any

import requests


class SlackNotifier:
    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def send(self, text: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        response = requests.post(
            self.webhook_url,
            json={"text": text, "blocks": blocks},
            timeout=30,
        )
        response.raise_for_status()
        return {"ok": True, "status_code": response.status_code}
