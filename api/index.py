"""Vercel entry point for the churn prediction API.

Re-exports the existing FastAPI app from src/api.py unchanged, with a
small ASGI wrapper that strips a leading /api prefix from the request
path (Vercel serves this function under /api) before the app's own
routes are matched. Requests without the prefix pass through untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import app as churn_app


class _StripApiPrefixMiddleware:
    """Strip a leading /api prefix from the request path, if present."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            if path.startswith("/api/"):
                scope["path"] = path[len("/api"):]
            elif path == "/api":
                scope["path"] = "/"
        await self.app(scope, receive, send)


app = _StripApiPrefixMiddleware(churn_app)


@churn_app.get("/{full_path:path}", include_in_schema=False)
async def _echo_seen_path(full_path: str):
    """TEMPORARY diagnostic: report the path the app actually sees."""
    return {"seen_path": "/" + full_path}
