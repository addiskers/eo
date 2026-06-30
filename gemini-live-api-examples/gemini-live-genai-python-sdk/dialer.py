"""
Single outbound-call entry point (Plivo).

Used by BOTH the /call-me admin button and the auto-callback scheduler, so the dial
logic lives in exactly one place. `place_call()` runs the blocking Plivo SDK call in a
thread executor and returns a structured dict — it never raises for normal API errors.
"""

import asyncio
import logging
import os
from urllib.parse import quote

logger = logging.getLogger(__name__)


def _base_url(base_url=None, request=None):
    """Resolve the public https base Plivo must reach for /plivo/answer."""
    public = (os.getenv("PUBLIC_URL", "") or "").rstrip("/")
    if public:
        return public
    if base_url:
        return base_url.rstrip("/")
    if request is not None:
        host = request.headers.get("host", "localhost")
        proto = ("https" if ("onrender.com" in host or "globalvoxinc.ai" in host)
                 else request.url.scheme)
        return f"{proto}://{host}"
    return ""


def _place_call_sync(to_number, answer_url):
    import plivo
    client = plivo.RestClient(os.getenv("PLIVO_AUTH_ID"), os.getenv("PLIVO_AUTH_TOKEN"))
    resp = client.calls.create(
        from_=os.getenv("PLIVO_FROM_NUMBER", ""),
        to_=to_number,
        answer_url=answer_url,
        answer_method="GET",
    )
    return getattr(resp, "request_uuid", None) or (
        resp.get("request_uuid") if isinstance(resp, dict) else None)


async def place_call(to_number, *, base_url=None, request=None, gen=0, origin_call_id=None):
    """
    Place one outbound Plivo call that bridges to the agent.

    Returns {"success": True, "call_uuid": ..., "to": ...} or {"error": "..."}.
    `gen` / `origin_call_id` are threaded into the answer_url so a re-dialed call
    knows it is a callback (used by the scheduler to cap callback generations).
    """
    if not to_number:
        return {"error": "Missing 'to' number"}
    if not os.getenv("PLIVO_AUTH_ID") or not os.getenv("PLIVO_AUTH_TOKEN"):
        return {"error": "Plivo credentials not configured"}
    if not os.getenv("PLIVO_FROM_NUMBER"):
        return {"error": "PLIVO_FROM_NUMBER not configured"}

    base = _base_url(base_url=base_url, request=request)
    if not base:
        return {"error": "No PUBLIC_URL configured; cannot build answer_url"}

    answer_url = f"{base}/plivo/answer?caller={quote(to_number)}"
    if gen:
        answer_url += f"&gen={int(gen)}"
    if origin_call_id:
        answer_url += f"&origin={quote(str(origin_call_id))}"

    try:
        loop = asyncio.get_running_loop()
        request_uuid = await loop.run_in_executor(
            None, _place_call_sync, to_number, answer_url)
        logger.info(f"Outbound Plivo call initiated: {request_uuid} to {to_number} (gen={gen})")
        return {"success": True, "call_uuid": request_uuid, "to": to_number}
    except Exception as e:
        logger.error(f"Failed to initiate Plivo call to {to_number}: {e}")
        return {"error": str(e)}
