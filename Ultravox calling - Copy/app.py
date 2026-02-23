import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from twilio.rest import Client
import httpx

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voice_agent")

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DEFAULT_TO_NUMBER = os.getenv("DEFAULT_TO_NUMBER", "+918281359250")
AUTO_CALL_ON_STARTUP = os.getenv("AUTO_CALL_ON_STARTUP", "true").lower() in ("true", "1", "yes")
_raw_webhook = os.getenv("WEBHOOK_URL", "").strip().rstrip("/")
WEBHOOK_URL = _raw_webhook if _raw_webhook.startswith("http") else (f"https://{_raw_webhook}" if _raw_webhook else "")

# Ultravox
ULTRAVOX_API_KEY = os.getenv("ULTRAVOX_API_KEY")
ULTRAVOX_AGENT_ID = os.getenv("ULTRAVOX_AGENT_ID")  # Optional: default agent UUID
ULTRAVOX_SYSTEM_PROMPT = os.getenv(
    "ULTRAVOX_SYSTEM_PROMPT",
    "You are a helpful voice assistant. Be concise and natural.",
)
ULTRAVOX_VOICE = os.getenv("ULTRAVOX_VOICE", "Mark")
ULTRAVOX_MODEL = os.getenv("ULTRAVOX_MODEL", "ultravox-v0.7")
# Lower latency: reduce time before agent responds (default 0.384s). Use 32ms multiples: 0.192s, 0.224s, 0.256s.
ULTRAVOX_TURN_ENDPOINT_DELAY = os.getenv("ULTRAVOX_TURN_ENDPOINT_DELAY", "0.224s")
ULTRAVOX_TEMPERATURE = float(os.getenv("ULTRAVOX_TEMPERATURE", "0.25"))



async def _trigger_outbound_call():
    """Trigger outbound call to DEFAULT_TO_NUMBER (used on startup if AUTO_CALL_ON_STARTUP)."""
    if not DEFAULT_TO_NUMBER or not ULTRAVOX_API_KEY:
        return
    ultravox_agent_id = ULTRAVOX_AGENT_ID if _is_agent_uuid(ULTRAVOX_AGENT_ID) else None
    try:
        if _is_agent_uuid(ultravox_agent_id):
            join_url, call_id = await _create_ultravox_agent_call(ultravox_agent_id, {})
        else:
            join_url, call_id = await _create_ultravox_direct_call(
                ULTRAVOX_SYSTEM_PROMPT, ULTRAVOX_VOICE, ULTRAVOX_MODEL
            )
        logger.info("=" * 50)
        logger.info("CALL STARTED (auto)")
        logger.info(f"  To: {DEFAULT_TO_NUMBER}")
        logger.info(f"  Ultravox callId: {call_id or 'n/a'}")
        twiml = f'<Response><Connect><Stream url="{join_url}"/></Connect></Response>'
        params = {"to": DEFAULT_TO_NUMBER, "from_": TWILIO_PHONE_NUMBER, "twiml": twiml}
        if WEBHOOK_URL:
            params["status_callback"] = f"{WEBHOOK_URL.rstrip('/')}/twilio/status"
            params["status_callback_event"] = ["initiated", "ringing", "answered", "completed"]
        call = twilio_client.calls.create(**params)
        logger.info(f"  Twilio SID: {call.sid}")
        logger.info("=" * 50)
    except Exception as e:
        logger.error(f"Auto-call failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if AUTO_CALL_ON_STARTUP:
        await asyncio.sleep(2)  # Wait for server to be ready
        asyncio.create_task(_trigger_outbound_call())
    yield


app = FastAPI(lifespan=lifespan)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
def _is_agent_uuid(value: str | None) -> bool:
    return bool(value and UUID_PATTERN.match(value.strip()))


@app.get("/")
async def root():
    base = WEBHOOK_URL or "http://localhost:8080"
    return {
        "message": "Twilio + Ultravox Outbound Call Server",
        "webhook_url": base if WEBHOOK_URL else None,
        "outbound_call": f"{base}/twilio/outbound_call" if WEBHOOK_URL else "POST /twilio/outbound_call",
        "body": {
            "to": "required",
            "agent_id": "optional - Ultravox agent UUID (or set ULTRAVOX_AGENT_ID in .env)",
            "dynamic_variables": {
                "template_context": "for agent templates, e.g. {customerName: 'John'}",
                "system_prompt": "override for direct calls",
                "voice": "override for direct calls",
                "model": "override for direct calls",
            },
        },
    }


def _vad_settings():
    """VAD settings for lower latency (faster agent response after user stops)."""
    return {"turnEndpointDelay": ULTRAVOX_TURN_ENDPOINT_DELAY}


def _ultravox_callbacks() -> dict:
    """Callbacks for call lifecycle (e.g. ended) so we can fetch and log transcript."""
    if not WEBHOOK_URL:
        return {}
    base = WEBHOOK_URL.rstrip("/")
    return {
        "ended": {"url": f"{base}/ultravox/call_ended"},
    }


async def _create_ultravox_agent_call(agent_id: str, template_context: dict) -> tuple[str, str | None]:
    """Create Ultravox call using an existing agent. Returns (joinUrl, callId).
    StartAgentCallRequest does not include vadSettings (only /api/calls does).
    """
    payload = {
        "medium": {"twilio": {}},
        "firstSpeakerSettings": {"user": {}},
    }
    if template_context:
        payload["templateContext"] = template_context
    callbacks = _ultravox_callbacks()
    if callbacks:
        payload["callbacks"] = callbacks

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"https://api.ultravox.ai/api/agents/{agent_id}/calls",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": ULTRAVOX_API_KEY,
            },
        )
        if r.status_code >= 400:
            try:
                err_body = r.json()
            except Exception:
                err_body = r.text
            logger.error(f"Ultravox agent call API error {r.status_code}: {err_body}")
        r.raise_for_status()
        data = r.json()
    join_url = data.get("joinUrl")
    if not join_url:
        raise ValueError("Ultravox response missing joinUrl")
    call_id = data.get("callId")
    return join_url, call_id


async def _create_ultravox_direct_call(
    system_prompt: str, voice: str, model: str
) -> tuple[str, str | None]:
    """Create Ultravox call with inline config. Returns (joinUrl, callId)."""
    payload = {
        "systemPrompt": system_prompt,
        "model": model,
        "voice": voice,
        "temperature": ULTRAVOX_TEMPERATURE,
        "firstSpeakerSettings": {"user": {}},
        "medium": {"twilio": {}},
        "vadSettings": _vad_settings(),
    }
    callbacks = _ultravox_callbacks()
    if callbacks:
        payload["callbacks"] = callbacks

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            "https://api.ultravox.ai/api/calls",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": ULTRAVOX_API_KEY,
            },
        )
        r.raise_for_status()
        data = r.json()
    join_url = data.get("joinUrl")
    if not join_url:
        raise ValueError("Ultravox response missing joinUrl")
    call_id = data.get("callId")
    return join_url, call_id


@app.post("/twilio/status")
async def twilio_status(request: Request):
    """Twilio status callback - receives call lifecycle events (initiated, ringing, answered, completed)."""
    try:
        form = await request.form()
        data = dict(form)
        call_sid = data.get("CallSid")
        call_status = (data.get("CallStatus") or "").strip().lower()
        status_labels = {
            "initiated": "Call initiated (dialing)",
            "ringing": "Ringing",
            "answered": "Answered – conversation started",
            "completed": "Call completed",
        }
        label = status_labels.get(call_status, call_status)
        logger.info(f"[Twilio] {label} | CallSid={call_sid}")
        return {"ok": True}
    except Exception as e:
        logger.error(f"Twilio status callback error: {e}")
        return {"ok": False}


async def _fetch_and_log_transcript(call_id: str) -> None:
    """Fetch call messages from Ultravox and log transcript (Agent: / User:) to terminal."""
    if not call_id or not ULTRAVOX_API_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"https://api.ultravox.ai/api/calls/{call_id}/messages",
                params={"mode": "in_call", "pageSize": 100},
                headers={"X-API-Key": ULTRAVOX_API_KEY},
            )
            r.raise_for_status()
            data = r.json()
        results = data.get("results") or []
        role_labels = {
            "MESSAGE_ROLE_USER": "User",
            "MESSAGE_ROLE_AGENT": "Agent",
            "MESSAGE_ROLE_TOOL_CALL": "Tool (call)",
            "MESSAGE_ROLE_TOOL_RESULT": "Tool (result)",
        }
        logger.info("")
        logger.info("-" * 50)
        logger.info(f"CALL TRANSCRIPT (callId={call_id})")
        logger.info("-" * 50)
        for msg in results:
            role = msg.get("role") or "MESSAGE_ROLE_UNSPECIFIED"
            text = (msg.get("text") or "").strip()
            if not text and role in ("MESSAGE_ROLE_TOOL_CALL", "MESSAGE_ROLE_TOOL_RESULT"):
                continue
            label = role_labels.get(role, role.replace("MESSAGE_ROLE_", "").title())
            logger.info(f"  {label}: {text or '(no text)'}")
        logger.info("-" * 50)
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to fetch transcript for call {call_id}: {e}")


@app.post("/ultravox/call_ended")
async def ultravox_call_ended(request: Request):
    """Ultravox lifecycle callback when a call ends. Fetches and logs transcript to terminal."""
    try:
        body = await request.json()
        call = body.get("call") or body
        call_id = call.get("callId") if isinstance(call, dict) else None
        if not call_id:
            logger.warning("Ultravox call_ended: no callId in payload")
            return {"ok": True}
        logger.info(f"[Ultravox] Call ended: {call_id}")
        await _fetch_and_log_transcript(call_id)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Ultravox call_ended callback error: {e}")
        return {"ok": True}


@app.post("/twilio/outbound_call")
async def outbound_call(request: Request):
    data = await request.json()
    to_number = data.get("to") or DEFAULT_TO_NUMBER
    agent_id = data.get("agent_id")
    dynamic_vars = data.get("dynamic_variables", {}) or {}
    logger.info(f"Outbound call to {to_number}, agent_id={agent_id}, dynamic_vars={dynamic_vars}")

    if not to_number:
        return {"error": "Missing 'to' number. Set in request body or DEFAULT_TO_NUMBER in .env"}
    if not ULTRAVOX_API_KEY:
        return {"error": "ULTRAVOX_API_KEY not set in .env"}

    # Use agent call if we have an Ultravox agent UUID (from request or env)
    ultravox_agent_id = agent_id if _is_agent_uuid(agent_id) else ULTRAVOX_AGENT_ID
    if _is_agent_uuid(ultravox_agent_id):
        template_context = dynamic_vars.get("template_context", dynamic_vars)
        if "system_prompt" in template_context or "voice" in template_context or "model" in template_context:
            template_context = {k: v for k, v in template_context.items() if k not in ("system_prompt", "voice", "model")}
        try:
            join_url, call_id = await _create_ultravox_agent_call(ultravox_agent_id, template_context)
        except httpx.HTTPStatusError as e:
            logger.error(f"Ultravox agent API error: {e.response.status_code} {e.response.text}")
            return {"error": f"Ultravox API error: {e.response.status_code}", "detail": e.response.text}
    else:
        system_prompt = dynamic_vars.get("system_prompt") or ULTRAVOX_SYSTEM_PROMPT
        voice = dynamic_vars.get("voice") or ULTRAVOX_VOICE
        model = dynamic_vars.get("model") or ULTRAVOX_MODEL
        try:
            join_url, call_id = await _create_ultravox_direct_call(system_prompt, voice, model)
        except httpx.HTTPStatusError as e:
            logger.error(f"Ultravox API error: {e.response.status_code} {e.response.text}")
            return {"error": f"Ultravox API error: {e.response.status_code}", "detail": e.response.text}

    logger.info("=" * 50)
    logger.info("CALL STARTED")
    logger.info(f"  To: {to_number}")
    logger.info(f"  Ultravox callId: {call_id or 'n/a'}")
    twiml = f'<Response><Connect><Stream url="{join_url}"/></Connect></Response>'
    create_params = {
        "to": to_number,
        "from_": TWILIO_PHONE_NUMBER,
        "twiml": twiml,
    }
    if WEBHOOK_URL:
        create_params["status_callback"] = f"{WEBHOOK_URL.rstrip('/')}/twilio/status"
        create_params["status_callback_event"] = ["initiated", "ringing", "answered", "completed"]

    try:
        call = twilio_client.calls.create(**create_params)
        logger.info(f"  Twilio SID: {call.sid}")
        logger.info("=" * 50)
        return {"success": True, "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Error creating Twilio call: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
