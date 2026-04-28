"""One-shot OAuth login helper for Specter MCP.

Run this once on a workstation to mint a refresh token for a shared service
account. Paste the printed token into Railway as `SPECTER_MCP_REFRESH_TOKEN`
on both the web and worker services in the testing project.

Specter MCP supports OAuth 2.1 with Dynamic Client Registration (RFC 7591) and
PKCE. The flow:
    1. Discover the authorization server metadata.
    2. Dynamically register a client.
    3. Open a browser to /authorize with PKCE.
    4. Capture the code on a local callback HTTP server.
    5. Exchange the code at /token for access + refresh tokens.

Usage::

    python scripts/specter_oauth_login.py

After running, the script prints the refresh token (do not commit it). Store
it as `SPECTER_MCP_REFRESH_TOKEN` and the issued client credentials as
`SPECTER_MCP_CLIENT_ID` / `SPECTER_MCP_CLIENT_SECRET` on Railway.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import socket
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from typing import Any

MCP_SERVER_URL = "https://mcp.tryspecter.com/mcp"
PROTECTED_RESOURCE_METADATA_URL = (
    "https://mcp.tryspecter.com/.well-known/oauth-protected-resource/mcp"
)
CALLBACK_HOST = "127.0.0.1"
CLIENT_NAME = "Rockaway Deal Intelligence Worker"


def _http_get_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed: HTTP {exc.code} {body}") from exc


def _http_post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST {url} failed: HTTP {exc.code} {body_text}"
        ) from exc


def _http_post_form(
    url: str, body: dict[str, str], auth: tuple[str, str] | None = None
) -> dict[str, Any]:
    data = urllib.parse.urlencode(body).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    if auth is not None:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST {url} failed: HTTP {exc.code} {body_text}"
        ) from exc


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((CALLBACK_HOST, 0))
        return sock.getsockname()[1]


def _make_pkce_pair() -> tuple[str, str]:
    verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    )
    challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    return verifier, challenge


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server convention
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_error(404)
            return
        params = dict(urllib.parse.parse_qsl(parsed.query))
        self.server.captured_params = params  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        body = b"<html><body><h2>Specter login complete.</h2>"
        body += b"<p>You can close this tab and return to the terminal.</p>"
        body += b"</body></html>"
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - http.server convention
        pass  # silence default access logs


def _await_callback(port: int, expected_state: str, timeout_sec: int = 300) -> dict[str, str]:
    server = http.server.HTTPServer((CALLBACK_HOST, port), _CallbackHandler)
    server.captured_params = None  # type: ignore[attr-defined]
    server.timeout = 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = timeout_sec * 10
        for _ in range(deadline):
            params = server.captured_params  # type: ignore[attr-defined]
            if params is not None:
                if params.get("state") != expected_state:
                    raise RuntimeError("State mismatch in OAuth callback")
                if "error" in params:
                    raise RuntimeError(
                        f"Authorization error: {params['error']} {params.get('error_description', '')}"
                    )
                if "code" not in params:
                    raise RuntimeError(f"No code in callback: {params!r}")
                return params
            thread.join(0.1)
        raise TimeoutError("OAuth callback not received in time")
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    print("Discovering Specter MCP authorization server...")
    pr_meta = _http_get_json(PROTECTED_RESOURCE_METADATA_URL)
    if not pr_meta.get("authorization_servers"):
        raise RuntimeError(
            "Protected resource metadata is missing authorization_servers"
        )
    issuer = pr_meta["authorization_servers"][0].rstrip("/")
    auth_meta = _http_get_json(f"{issuer}/.well-known/oauth-authorization-server")
    print(f"  authorization_endpoint = {auth_meta['authorization_endpoint']}")
    print(f"  token_endpoint         = {auth_meta['token_endpoint']}")
    print(f"  registration_endpoint  = {auth_meta['registration_endpoint']}")

    callback_port = _pick_free_port()
    redirect_uri = f"http://{CALLBACK_HOST}:{callback_port}/callback"

    print("\nRegistering client via Dynamic Client Registration...")
    # Specter's authorization-server metadata declares an empty
    # `scopes_supported` list, so we don't request any scope at DCR time.
    # Refresh tokens are still issued because we ask for the `refresh_token`
    # grant type explicitly.
    reg = _http_post_json(
        auth_meta["registration_endpoint"],
        {
            "client_name": CLIENT_NAME,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        },
    )
    client_id = reg["client_id"]
    client_secret = reg.get("client_secret")
    print(f"  client_id = {client_id}")

    verifier, challenge = _make_pkce_pair()
    state = secrets.token_urlsafe(24)
    auth_query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    auth_url = f"{auth_meta['authorization_endpoint']}?{auth_query}"

    print("\nOpening your browser to complete Specter SSO...")
    print(f"If the browser does not open, paste this URL: {auth_url}\n")
    webbrowser.open(auth_url)

    params = _await_callback(callback_port, expected_state=state)
    code = params["code"]

    print("Exchanging code for tokens...")
    token_body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    auth = (client_id, client_secret) if client_secret else None
    if auth is None:
        token_body["client_id"] = client_id
    tokens = _http_post_form(auth_meta["token_endpoint"], token_body, auth=auth)

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        print(
            "WARNING: no refresh_token returned. Check that offline_access is "
            "supported by the issuer.",
            file=sys.stderr,
        )

    print("\n=== Specter MCP credentials ===")
    print(f"SPECTER_MCP_URL={MCP_SERVER_URL}")
    print(f"SPECTER_MCP_CLIENT_ID={client_id}")
    if client_secret:
        print(f"SPECTER_MCP_CLIENT_SECRET={client_secret}")
    if refresh_token:
        print(f"SPECTER_MCP_REFRESH_TOKEN={refresh_token}")
    print(f"SPECTER_MCP_TOKEN_ENDPOINT={auth_meta['token_endpoint']}")
    print()
    print("Set these on the Railway testing project (web + worker services):")
    print("  railway variables --set 'SPECTER_MCP_URL=...' \\")
    print("                    --set 'SPECTER_MCP_CLIENT_ID=...' \\")
    print("                    --set 'SPECTER_MCP_CLIENT_SECRET=...' \\")
    print("                    --set 'SPECTER_MCP_REFRESH_TOKEN=...' \\")
    print("                    --set 'SPECTER_MCP_TOKEN_ENDPOINT=...'")
    print()
    print("Treat the refresh token like a password. Re-run this script when")
    print("the token expires (Specter session has a bounded lifetime).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
