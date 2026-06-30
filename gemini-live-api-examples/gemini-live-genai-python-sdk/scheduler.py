"""
In-process auto-callback scheduler.

A single asyncio task (the app runs ONE uvicorn worker) polls the call store for
`callback` blocks that are due, atomically claims each one (flip to `in_flight` and
persist BEFORE dialing), and re-dials via dialer.place_call().

Safety properties:
- Atomic claim + single worker  → no double-fire. Crash-mid-claim is healed on boot by
  store.reset_orphaned_callbacks(), which uses result_call_id to tell "already dialed"
  from "never dialed".
- Retries with exponential backoff, capped by max_attempts (gated on next_retry_at;
  due_at is immutable).
- Durable circuit breaker (a persisted `paused_until`): after N consecutive dial errors
  it pauses dialing for a cooldown and SURVIVES restarts, so a redeploy loop can't
  re-burst the carrier. A single success clears it → self-heals once outbound is
  unblocked.
- Master switch CALLBACK_SCHEDULER_ENABLED (env) plus an in-memory admin override.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import dialer
import store

logger = logging.getLogger(__name__)

# In-memory admin override (None = follow env). Set via the admin toggle endpoint.
_enabled_override = None
# Consecutive dial-failure counter (the resulting pause is persisted in the store).
_consecutive_failures = 0


def _cfg_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def _parse(s):
    try:
        return datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def set_override(value):
    """True/False to force on/off, None to follow the env var."""
    global _enabled_override
    _enabled_override = value


def is_enabled():
    if _enabled_override is not None:
        return _enabled_override
    return os.getenv("CALLBACK_SCHEDULER_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off")


def _backoff_seconds(attempts):
    base = _cfg_int("CALLBACK_BACKOFF_BASE", 300)
    cap = _cfg_int("CALLBACK_BACKOFF_MAX", 3600)
    return min(base * (2 ** max(0, attempts - 1)), cap)


async def _is_paused(now):
    state = await store.load_scheduler_state()
    pu = _parse(state.get("paused_until"))
    return pu is not None and pu > now


async def _trip_breaker(now):
    until = _iso(now + timedelta(seconds=_cfg_int("CALLBACK_FAILFAST_COOLDOWN", 1800)))
    state = await store.load_scheduler_state()
    state["paused_until"] = until
    await store.save_scheduler_state(state)
    logger.warning(f"Callback scheduler circuit breaker tripped; paused until {until}")


async def _clear_breaker():
    state = await store.load_scheduler_state()
    if state.get("paused_until"):
        state["paused_until"] = None
        await store.save_scheduler_state(state)


async def _settle(call, cb, res, now):
    global _consecutive_failures
    if res.get("success"):
        cb["status"] = "completed"
        cb["result_call_id"] = res.get("call_uuid")
        cb["next_retry_at"] = None
        cb["last_error"] = None
        _consecutive_failures = 0
        await _clear_breaker()
    else:
        cb["last_error"] = res.get("error") or "unknown error"
        _consecutive_failures += 1
        if cb.get("attempts", 0) >= cb.get("max_attempts", 3):
            cb["status"] = "failed"
            cb["next_retry_at"] = None
        else:
            cb["status"] = "pending"
            cb["next_retry_at"] = _iso(now + timedelta(seconds=_backoff_seconds(cb["attempts"])))
        if _consecutive_failures >= _cfg_int("CALLBACK_FAILFAST_THRESHOLD", 5):
            await _trip_breaker(now)
    await store.save_call(call)


async def _tick():
    now = _now()
    if await _is_paused(now):
        return
    max_per_tick = _cfg_int("CALLBACK_MAX_PER_TICK", 5)
    due = await store.list_pending_callbacks(_iso(now))
    dialed = 0
    for meta in due:
        if dialed >= max_per_tick:
            break
        if await _is_paused(_now()):          # breaker may trip mid-loop
            break
        # Re-load the live record and re-check status before claiming.
        call = await store.load_call(meta["id"])
        cb = (call or {}).get("callback")
        if not cb or cb.get("status") != "pending":
            continue
        nr = _parse(cb.get("next_retry_at"))
        if nr is not None and nr > _now():
            continue
        to = cb.get("to")
        if not to:
            cb["status"] = "failed"
            cb["last_error"] = "no destination number"
            await store.save_call(call)
            continue
        # ---- ATOMIC CLAIM: flip + SAVE before dialing ----
        claim_now = _now()
        cb["status"] = "in_flight"
        cb["attempts"] = cb.get("attempts", 0) + 1
        cb["last_attempt_at"] = _iso(claim_now)
        await store.save_call(call)
        dialed += 1
        res = await dialer.place_call(
            to,
            base_url=os.getenv("PUBLIC_URL"),
            gen=int(cb.get("generation", 1)),
            origin_call_id=cb.get("origin_call_id") or call.get("id"),
        )
        await _settle(call, cb, res, _now())


async def run_loop():
    interval = _cfg_int("CALLBACK_POLL_INTERVAL", 30)
    logger.info(f"Callback scheduler started (interval={interval}s, enabled={is_enabled()})")
    while True:
        try:
            if is_enabled():
                await _tick()
        except asyncio.CancelledError:
            logger.info("Callback scheduler loop cancelled")
            raise
        except Exception as e:
            logger.warning(f"Callback scheduler tick error: {e}")
        await asyncio.sleep(interval)
