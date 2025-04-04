
"""
Utility function to convert UTC time to JST (Japan Standard Time).
"""

from datetime import datetime
from zoneinfo import ZoneInfo


def from_utc_to_jst(date_string: str) -> datetime:
    """
    Convert UTC time string to JST (Japan Standard Time).

    Args:
        date_string           : UTC time string
    Returns:
        datetime              : Datetime converted to JST
    """
    date_utc = datetime.fromisoformat(date_string.rstrip("Z")).replace(
        tzinfo=ZoneInfo("UTC")
    )
    return date_utc.astimezone(ZoneInfo("Asia/Tokyo"))
