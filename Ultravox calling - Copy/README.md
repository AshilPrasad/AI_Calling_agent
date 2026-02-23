# Twilio + Ultravox Outbound Calls

Outbound voice AI calls using **Ultravox** (end-to-end speech-to-speech). Twilio connects the call; Ultravox handles the conversation.

## Environment variables

Copy `.env.example` to `.env` and set:

- **Twilio**: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`
- **Ultravox**: `ULTRAVOX_API_KEY` (from [Ultravox](https://ultravox.ai))
- Optional: `ULTRAVOX_SYSTEM_PROMPT`, `ULTRAVOX_VOICE` (e.g. `Mark`), `ULTRAVOX_MODEL` (e.g. `ultravox-v0.7`)

## API

**POST /twilio/outbound_call**

```json
{
  "to": "+1234567890",
  "agent_id": "optional-identifier",
  "dynamic_variables": {
    "system_prompt": "Optional per-call prompt.",
    "voice": "Mark",
    "model": "ultravox-v0.7"
  }
}
```

`agent_id` is optional (for logging). Override prompt/voice/model per call via `dynamic_variables`.

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Trigger a call:

```bash
curl -X POST http://localhost:8080/twilio/outbound_call \
  -H "Content-Type: application/json" \
  -d '{"to": "+1234567890", "agent_id": "test", "dynamic_variables": {}}'
```
