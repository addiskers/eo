"""
Callback scheduling helpers — pure functions, stdlib only (zoneinfo).

`compute_due_at()` turns the agent-supplied callback time (an ISO-8601 string and/or
free text spoken by the member) into a concrete UTC due-time, with sane clamping.
`new_callback_record()` builds the `callback` block stored on a call record.

No I/O, no Plivo, no app imports — safe to unit-test in isolation. Times are parsed
in CALLBACK_TZ (default Asia/Kolkata) and always returned as ISO-8601 UTC.
"""

import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def _cfg_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _tz():
    try:
        return ZoneInfo(os.getenv("CALLBACK_TZ", "Asia/Kolkata"))
    except Exception:
        return ZoneInfo("Asia/Kolkata")


def _now_utc():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
_DAYPARTS = {"morning": (10, 0), "afternoon": (15, 0), "evening": (18, 0),
             "tonight": (20, 0), "night": (20, 0)}


def _parse_clock(text):
    """Return (hour, minute) from 'H', 'H:MM' with optional am/pm; else None."""
    m = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', text)
    if m:
        h = int(m.group(1)) % 12
        if m.group(3) == "pm":
            h += 12
        return h, int(m.group(2) or 0)
    m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h, int(m.group(2))
    return None


def _parse_text(text, now_local):
    """Best-effort: turn spoken text into a localized datetime, or None."""
    t = (text or "").lower().strip()
    if not t:
        return None

    # "in N minutes/hours/days"
    m = re.search(r'\bin\s+(\d{1,3})\s*(minute|min|hour|hr|h|day|d)s?\b', t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith("d"):
            return now_local + timedelta(days=n)
        if unit in ("h", "hr", "hour"):
            return now_local + timedelta(hours=n)
        return now_local + timedelta(minutes=n)

    clock = _parse_clock(t)

    base = None
    if "day after tomorrow" in t:
        base = now_local + timedelta(days=2)
    elif "tomorrow" in t:
        base = now_local + timedelta(days=1)
    elif "tonight" in t:
        base = now_local
        clock = clock or _DAYPARTS["tonight"]
    elif "today" in t:
        base = now_local

    if base is None:
        for name, wd in _WEEKDAYS.items():
            if name in t:
                days = (wd - now_local.weekday()) % 7 or 7  # next occurrence
                base = now_local + timedelta(days=days)
                break

    if clock is None:
        for name, hm in _DAYPARTS.items():
            if name in t:
                clock = hm
                break

    if base is None and clock is None:
        return None
    if base is None:
        base = now_local

    h, mnt = clock if clock else (10, 0)
    cand = base.replace(hour=h, minute=mnt, second=0, microsecond=0)
    # A bare time for today that already passed → push to tomorrow.
    if cand <= now_local and "tomorrow" not in t and base.date() == now_local.date():
        cand = cand + timedelta(days=1)
    return cand


def compute_due_at(callback_time_iso, callback_time_text):
    """
    Resolve a callback due-time.

    Returns (due_at_utc_iso, due_source) where due_source is "iso" | "text" | "default".
    Priority: a valid future agent ISO → parsed spoken text → default offset.
    Always clamped to [now + MIN_DELAY, now + MAX_HORIZON].
    """
    tz = _tz()
    now_utc = _now_utc()
    now_local = now_utc.astimezone(tz)
    min_dt = now_utc + timedelta(seconds=_cfg_int("CALLBACK_MIN_DELAY", 60))
    max_dt = now_utc + timedelta(seconds=_cfg_int("CALLBACK_MAX_HORIZON", 604800))

    due, source = None, "default"

    iso = (callback_time_iso or "").strip()
    if iso:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            dt = dt.astimezone(timezone.utc)
            if dt > now_utc:                      # reject past / present
                due, source = dt, "iso"
        except (ValueError, TypeError):
            due = None

    if due is None:
        local = _parse_text(callback_time_text, now_local)
        if local is not None:
            dt = local.astimezone(timezone.utc)
            if dt > now_utc:
                due, source = dt, "text"

    if due is None:
        due = now_utc + timedelta(seconds=_cfg_int("CALLBACK_DEFAULT_OFFSET", 7200))
        source = "default"

    if due < min_dt:
        due = min_dt
    if due > max_dt:
        due = max_dt
    return _iso(due), source


def new_callback_record(*, to, due_at, source_text, due_source,
                        origin_call_id, generation=0):
    """Build the `callback` block stored on a call record."""
    return {
        "status": "pending",          # pending|in_flight|completed|failed|cancelled
        "due_at": due_at,             # immutable ISO-8601 UTC
        "to": to,                     # phone number to dial (= call["caller"])
        "source_text": source_text or "",
        "due_source": due_source,     # iso|text|default
        "attempts": 0,
        "max_attempts": _cfg_int("CALLBACK_MAX_ATTEMPTS", 3),
        "last_error": None,
        "last_attempt_at": None,
        "next_retry_at": None,        # gate for backoff; due_at stays immutable
        "created_at": _iso(_now_utc()),
        "result_call_id": None,       # Plivo request_uuid once dialed
        "generation": int(generation or 0),
        "origin_call_id": origin_call_id,
    }
