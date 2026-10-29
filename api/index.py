"""Vercel entry point for the churn prediction API.

Re-exports the existing FastAPI app from src/api.py unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import app

__all__ = ["app"]
