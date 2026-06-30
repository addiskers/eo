"""
CallRecorder — taps the live event stream of a single call, accumulates the
transcript + real token usage, computes cost, and persists via store.py.

One instance per call. It is fed the SAME events that already drive the live
viewer, so it never changes call behavior. Every persistence call is guarded so
a storage failure can never break an in-progress call.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

import pricing
import store

logger = logging.getLogger(__name__)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _in_bucket(mod):
    m = str(mod).upper()
    if "AUDIO" in m:
        return "audio_in"
    if "IMAGE" in m or "VIDEO" in m:
        return "imgvid_in"
    return "text_in"


def _out_bucket(mod):
    m = str(mod).upper()
    if "AUDIO" in m:
        return "audio_out"
    return "text_out"


def _infer_language(text):
    """Best-effort language guess from the first user turn (Unicode blocks)."""
    if not text:
        return None
    gu = hi = latin = 0
    for ch in text:
        o = ord(ch)
        if 0x0A80 <= o <= 0x0AFF:
            gu += 1
        elif 0x0900 <= o <= 0x097F:
            hi += 1
        elif ("a" <= ch.lower() <= "z"):
            latin += 1
    if gu and gu >= hi:
        return "gu"
    if hi:
        return "hi"          # Devanagari (Hindi/Marathi share the script)
    if latin:
        return "en"
    return "unknown"


class CallRecorder:
    def __init__(self, model=None):
        self.model = model
        self.call = None          # the persisted dict (None until open)
        self._cur_role = None     # 'user' | 'gemini' currently buffering
        self._cur_text = []
        self._usage = []          # list of per-event usage snapshots
        self._lang_decided = False
        self._closed = False
        self._started_ts = None

    # ---- lifecycle ---------------------------------------------------------

    async def open(self, source, call_sid=None, caller=None, generation=0):
        try:
            call_id = uuid.uuid4().hex[:16]
            if not call_sid:
                call_sid = "web-" + uuid.uuid4().hex[:12]
            self._started_ts = datetime.now(timezone.utc)
            self.call = {
                "id": call_id,
                "call_sid": call_sid,
                "source": source,                 # 'plivo' | 'browser'
                "caller": caller,
                "generation": int(generation or 0),  # 0=original, n=nth auto-callback redial
                "started_at": self._started_ts.isoformat(),
                "ended_at": None,
                "duration_seconds": 0,
                "language": None,
                "status": "in_progress",
                "booking_created": False,
                "gemini_model": self.model,
                "tokens": pricing._empty_tokens(),
                "gemini_cost_usd": 0.0,
                "twilio": {"price_usd": None, "price_unit": None,
                           "status": None, "duration_seconds": None},
                "total_cost_usd": 0.0,
                "cost_estimated": False,
                "transcript": [],
                "tool_calls": [],
            }
            await store.save_call(self.call)
            logger.info(f"Recording call {call_id} ({source}, sid={call_sid})")
        except Exception as e:
            logger.warning(f"CallRecorder.open failed: {e}")
            self.call = None

    async def on_event(self, event):
        if self.call is None:
            return
        try:
            etype = event.get("type")
            if etype in ("user", "gemini"):
                self._accumulate_turn(etype, event.get("text", ""))
            elif etype == "tool_call":
                self._flush_turn()
                self._record_tool(event)
            elif etype == "usage":
                self._accumulate_usage(event)
            elif etype in ("turn_complete", "interrupted"):
                self._flush_turn()
        except Exception as e:
            logger.warning(f"CallRecorder.on_event failed: {e}")

    async def close(self, status="completed"):
        if self.call is None or self._closed:
            return
        self._closed = True
        try:
            self._flush_turn()
            ended = datetime.now(timezone.utc)
            self.call["ended_at"] = ended.isoformat()
            if self._started_ts:
                self.call["duration_seconds"] = max(0, int((ended - self._started_ts).total_seconds()))
            if self.call.get("status") == "in_progress":
                self.call["status"] = status

            self.call["tokens"] = self._finalize_tokens()
            self.call["gemini_cost_usd"] = pricing.compute_gemini_cost(self.call["tokens"])
            total, estimated = pricing.compute_total(self.call)
            self.call["total_cost_usd"] = total
            self.call["cost_estimated"] = estimated

            await store.save_call(self.call)
            logger.info(
                f"Call {self.call['id']} closed: {self.call['duration_seconds']}s, "
                f"gemini=${self.call['gemini_cost_usd']:.6f}, total=${total:.6f}"
            )

            if self.call["source"] == "twilio" and not str(self.call["call_sid"]).startswith("web-"):
                asyncio.create_task(self._deferred_twilio_refresh(self.call["id"], self.call["call_sid"]))
        except Exception as e:
            logger.warning(f"CallRecorder.close failed: {e}")

    # ---- internals ---------------------------------------------------------

    def _accumulate_turn(self, role, text):
        if not text:
            return
        if self._cur_role and self._cur_role != role:
            self._flush_turn()
        self._cur_role = role
        self._cur_text.append(text)

    def _flush_turn(self):
        if not self._cur_role or not self._cur_text:
            self._cur_role = None
            self._cur_text = []
            return
        text = "".join(self._cur_text).strip()
        if text:
            self.call["transcript"].append(
                {"role": self._cur_role, "text": text, "ts": _now_iso()}
            )
            if not self._lang_decided and self._cur_role == "user":
                lang = _infer_language(text)
                if lang:
                    self.call["language"] = lang
                    self._lang_decided = True
        self._cur_role = None
        self._cur_text = []

    def _record_tool(self, event):
        name = event.get("name")
        result = event.get("result")
        self.call["tool_calls"].append({
            "name": name,
            "args": event.get("args"),
            "result": result,
            "ts": _now_iso(),
        })
        if name == "record_rsvp" and isinstance(result, dict):
            status = result.get("outcome_status") or ("yes" if result.get("attending") else "no")
            # Reuse the existing booking_created flag so the admin dashboard keeps working.
            if result.get("attending") or status == "yes":
                self.call["booking_created"] = True
            self.call["rsvp_outcome_status"] = status
            self.call["rsvp_callback_time_text"] = result.get("callback_time_text", "") or ""
            self.call["rsvp_do_not_contact"] = bool(result.get("do_not_contact"))
            self.call["rsvp_accompanying_children"] = result.get("accompanying_children", "") or ""
            if status == "callback":
                self._schedule_callback(result)
            else:
                # Outcome changed (e.g. maybe→yes): cancel any callback we'd queued.
                cb = self.call.get("callback")
                if cb and cb.get("status") == "pending":
                    cb["status"] = "cancelled"

    def _schedule_callback(self, result):
        """Initialise the call's `callback` block; the scheduler picks it up after
        close() persists it. Sets the dict only — never spawns a task here (sync)."""
        try:
            import callbacks
            to = (self.call.get("caller") or "").strip()
            if not to:
                logger.info("Callback requested but no caller number; not scheduling")
                return
            cur_gen = int(self.call.get("generation") or 0)
            try:
                max_gen = int(os.getenv("CALLBACK_MAX_GENERATIONS", "1"))
            except ValueError:
                max_gen = 1
            if cur_gen >= max_gen:
                logger.info(f"Callback generation cap reached (gen={cur_gen}); not re-scheduling")
                return
            due_at, due_source = callbacks.compute_due_at(
                result.get("callback_time_iso"),
                result.get("callback_time_text") or result.get("note") or "")
            self.call["callback"] = callbacks.new_callback_record(
                to=to,
                due_at=due_at,
                source_text=result.get("callback_time_text") or "",
                due_source=due_source,
                origin_call_id=self.call.get("id"),
                generation=cur_gen + 1,
            )
            logger.info(f"Scheduled callback for {to} at {due_at} (source={due_source}, gen={cur_gen + 1})")
        except Exception as e:
            logger.warning(f"Failed to schedule callback: {e}")

    def _accumulate_usage(self, event):
        snap = {
            "total": int(event.get("total") or 0),
            "thoughts": int(event.get("thoughts") or 0),
            "in": {}, "out": {},
        }
        for mod, cnt in (event.get("prompt_by_modality") or []):
            b = _in_bucket(mod)
            snap["in"][b] = snap["in"].get(b, 0) + int(cnt or 0)
        for mod, cnt in (event.get("response_by_modality") or []):
            b = _out_bucket(mod)
            snap["out"][b] = snap["out"].get(b, 0) + int(cnt or 0)
        self._usage.append(snap)

    def _finalize_tokens(self):
        """
        Reconcile per-event usage snapshots into final token buckets.

        The Live API may report usage as per-turn increments OR as a running
        session total. A cumulative series is strictly non-decreasing, so if the
        `total` sequence never drops we take the LAST snapshot (the grand total);
        otherwise we SUM the increments.
        """
        tokens = pricing._empty_tokens()
        if not self._usage:
            return tokens

        totals = [e["total"] for e in self._usage]
        non_decreasing = all(totals[i] <= totals[i + 1] for i in range(len(totals) - 1))
        cumulative = non_decreasing and len(self._usage) > 1 and totals[-1] > 0

        if cumulative:
            chosen = [self._usage[-1]]
        else:
            chosen = self._usage

        for snap in chosen:
            for b, c in snap["in"].items():
                tokens[b] += c
            for b, c in snap["out"].items():
                tokens[b] += c
            tokens["thoughts"] += snap["thoughts"]

        bucket_sum = (tokens["text_in"] + tokens["audio_in"] + tokens["imgvid_in"]
                      + tokens["text_out"] + tokens["audio_out"] + tokens["thoughts"])
        reported_total = totals[-1] if cumulative else sum(totals)
        tokens["total"] = max(bucket_sum, reported_total)
        return tokens

    async def _deferred_twilio_refresh(self, call_id, call_sid):
        """Fetch Twilio's real billed price after the call ends (price lags)."""
        loop = asyncio.get_running_loop()
        for delay in (15, 45, 120):
            await asyncio.sleep(delay)
            try:
                info = await loop.run_in_executor(None, pricing.fetch_twilio_price, call_sid)
            except Exception as e:
                logger.warning(f"Twilio refresh error for {call_sid}: {e}")
                info = None
            if not info:
                continue

            call = await store.load_call(call_id)
            if not call:
                return
            call["twilio"].update({
                "price_unit": info.get("price_unit"),
                "status": info.get("status"),
                "duration_seconds": info.get("duration_seconds"),
            })
            if info.get("price_usd") is not None:
                call["twilio"]["price_usd"] = info["price_usd"]
                if info.get("duration_seconds"):
                    call["duration_seconds"] = info["duration_seconds"]
                total, estimated = pricing.compute_total(call)
                call["total_cost_usd"] = total
                call["cost_estimated"] = estimated
                await store.save_call(call)
                logger.info(f"Twilio price set for {call_id}: ${info['price_usd']}")
                return
            # status known but price still pending -> persist status, keep retrying
            await store.save_call(call)
        logger.info(f"Twilio price still unavailable for {call_id} after retries")
