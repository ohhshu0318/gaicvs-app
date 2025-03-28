from datetime import datetime
from zoneinfo import ZoneInfo


def from_utc_to_jst(date_string: str) -> datetime:
    date_utc = datetime.fromisoformat(date_string.rstrip("Z")).replace(
        tzinfo=ZoneInfo("UTC")
    )
    return date_utc.astimezone(ZoneInfo("Asia/Tokyo"))
