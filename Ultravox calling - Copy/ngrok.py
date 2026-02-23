import os
import sys

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

NGROK_API = "http://127.0.0.1:4040/api/tunnels"
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def get_url() -> str:
    """Fetch ngrok HTTPS URL and update .env. Returns the domain (without https://)."""
    try:
        res = requests.get(NGROK_API, timeout=5)
        res.raise_for_status()
        data = res.json()
    except requests.RequestException as e:
        print(f"Error: Could not reach ngrok API at {NGROK_API}")
        print("Make sure ngrok is running: ngrok http 8080")
        sys.exit(1)

    tunnels = data.get("tunnels", [])
    https_url = None
    for t in tunnels:
        url = t.get("public_url", "")
        if url.startswith("https://"):
            https_url = url
            break

    if not https_url:
        print("Error: No HTTPS tunnel found. Run: ngrok http 8080")
        sys.exit(1)

    domain = https_url.replace("https://", "").rstrip("/")
    _update_env(domain)
    print(f"Updated .env with WEBHOOK_URL={domain}")
    return domain


def _update_env(domain: str) -> None:
    """Add or update WEBHOOK_URL in .env."""
    if not os.path.exists(ENV_PATH):
        with open(ENV_PATH, "w") as f:
            f.write(f"WEBHOOK_URL={domain}\n")
        return

    with open(ENV_PATH, "r") as f:
        lines = f.readlines()

    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("WEBHOOK_URL="):
            new_lines.append(f"WEBHOOK_URL={domain}\n")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"\n# Webhook base URL (ngrok or public host)\nWEBHOOK_URL={domain}\n")

    with open(ENV_PATH, "w") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    get_url()
