from __future__ import annotations

from typing import Any

import resend


class ReportEmailer:
    def __init__(self, api_key: str, from_email: str) -> None:
        resend.api_key = api_key
        self.from_email = from_email

    def send(self, to_emails: list[str], subject: str, html: str) -> dict[str, Any]:
        params: resend.Emails.SendParams = {
            "from": self.from_email,
            "to": to_emails,
            "subject": subject,
            "html": html,
        }
        return resend.Emails.send(params)
