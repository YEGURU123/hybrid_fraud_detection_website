"""
Small helper utilities shared across the fraud_detection package.
(This file was not included in the original code dump, so a minimal
version is provided here so `from fraud_detection import utils` works
if you extend the project later.)
"""

from datetime import datetime


def format_currency(amount: float) -> str:
    """Format a number as a USD currency string."""
    return f"${amount:,.2f}"


def timestamp_now() -> str:
    """Return an ISO-8601 timestamp for the current moment."""
    return datetime.now().isoformat()


def risk_color(risk_level: str) -> str:
    """Map a risk level string to a Bootstrap color class, used by the UI."""
    mapping = {
        'Low': 'success',
        'Medium': 'warning',
        'High': 'danger',
        'Critical': 'dark',
    }
    return mapping.get(risk_level, 'secondary')
