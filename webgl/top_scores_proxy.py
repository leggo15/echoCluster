#!/usr/bin/env python3
"""Local proxy that exposes osu! user top-100 beatmap IDs to the WebGL viewer.

The viewer's "Top Scores" search mode sends requests to this process.
GitHub Pages deployments cannot reach localhost, so the search type is
greyed out there and only enabled when this script is detected running.

Usage
-----
    python top_scores_proxy.py --client-id ID --client-secret SECRET [--port 7373]

Get your client_id / client_secret at:
    https://osu.ppy.sh/home/account/edit  →  "OAuth"  →  "New OAuth Application"
    (Application Callback URL can be set to http://localhost — it is not used here)

The script uses only the Python standard library — no pip installs needed.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ── module-level state ────────────────────────────────────────────────────────
_CLIENT_ID:     str   = ""
_CLIENT_SECRET: str   = ""
_PORT:          int   = 7373

OSU_TOKEN_URL = "https://osu.ppy.sh/oauth/token"
OSU_API_BASE  = "https://osu.ppy.sh/api/v2"

_access_token:   str   = ""
_token_expires:  float = 0.0   # unix timestamp after which the token must be refreshed


# ── OAuth helpers ─────────────────────────────────────────────────────────────

def _get_token() -> str:
    """Return a valid Bearer token, refreshing it if necessary."""
    global _access_token, _token_expires
    if _access_token and time.time() < _token_expires - 60:
        return _access_token

    print("[top_scores_proxy] obtaining OAuth token …")
    payload = json.dumps({
        "client_id":     int(_CLIENT_ID),
        "client_secret": _CLIENT_SECRET,
        "grant_type":    "client_credentials",
        "scope":         "public",
    }).encode("utf-8")
    req = urllib.request.Request(
        OSU_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    _access_token  = data["access_token"]
    _token_expires = time.time() + int(data.get("expires_in", 86400))
    print(f"[top_scores_proxy] token obtained (expires in {int(data.get('expires_in', 86400))}s)")
    return _access_token


def _api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET {OSU_API_BASE}{path}?{params} with a valid Bearer token."""
    token = _get_token()
    url   = OSU_API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self) -> None:
        self._send_cors(200, b"")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)

        if parsed.path in ("/ping", "/ping/"):
            self._json({"status": "ok", "port": _PORT})
            return

        action = (params.get("action") or [""])[0]
        if action != "top100":
            self._json({"error": f"Unknown action '{action}'"}, code=400)
            return

        user = ((params.get("user") or [""])[0]).strip()
        if not user:
            self._json({"error": "Missing 'user' parameter"}, code=400)
            return

        limit = min(int((params.get("limit") or ["100"])[0]), 100)

        try:
            # ── 1. Resolve username → numeric user ID if needed ──────────────
            if user.isdigit():
                user_id = user
            else:
                print(f"[top_scores_proxy] resolving username '{user}' …")
                user_data = _api_get(f"/users/{urllib.parse.quote(user)}/osu",
                                     {"key": "username"})
                user_id = str(user_data["id"])
                print(f"[top_scores_proxy] resolved '{user}' → user_id={user_id}")

            # ── 2. Fetch top scores ──────────────────────────────────────────
            print(f"[top_scores_proxy] fetching top {limit} scores for user_id={user_id}")
            scores = _api_get(
                f"/users/{user_id}/scores/best",
                {"limit": limit, "ruleset": "osu"},
            )

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg  = f"osu! API HTTP {exc.code}: {exc.reason} — {body[:300]}"
            print(f"[top_scores_proxy] ERROR {msg}")
            self._json({"error": msg}, code=502)
            return
        except Exception as exc:
            msg = str(exc)
            print(f"[top_scores_proxy] ERROR {msg}")
            self._json({"error": msg}, code=502)
            return

        if not isinstance(scores, list):
            msg = f"Unexpected response from osu! API: {str(scores)[:300]}"
            print(f"[top_scores_proxy] ERROR {msg}")
            self._json({"error": msg}, code=502)
            return

        # v2 score objects nest the beatmap id under score["beatmap"]["id"]
        map_ids = [
            str(s["beatmap"]["id"])
            for s in scores
            if isinstance(s.get("beatmap"), dict) and s["beatmap"].get("id")
        ]
        print(f"[top_scores_proxy] returning {len(map_ids)} beatmap IDs")
        self._json({"user": user, "user_id": user_id, "count": len(map_ids), "map_ids": map_ids})

    # ── helpers ───────────────────────────────────────────────────────────────

    def _json(self, obj: object, code: int = 200) -> None:
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self._send_cors(code, body, content_type="application/json")

    def _send_cors(self, code: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
        print(f"[top_scores_proxy] {fmt % args}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _CLIENT_ID, _CLIENT_SECRET, _PORT

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--client-id",
        required=True,
        metavar="ID",
        help="osu! OAuth application client_id",
    )
    ap.add_argument(
        "--client-secret",
        required=True,
        metavar="SECRET",
        help="osu! OAuth application client_secret",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=7373,
        metavar="PORT",
        help="Port to listen on (default: 7373)",
    )
    args = ap.parse_args()

    _CLIENT_ID     = str(args.client_id).strip()
    _CLIENT_SECRET = str(args.client_secret).strip()
    _PORT          = int(args.port)

    # Eagerly fetch a token to validate credentials before accepting requests.
    try:
        _get_token()
    except Exception as exc:
        print(f"[top_scores_proxy] FATAL: could not obtain OAuth token: {exc}")
        raise SystemExit(1)

    print(f"[top_scores_proxy] listening on http://localhost:{_PORT}")
    print(f"[top_scores_proxy] open the WebGL viewer and select 'Top Scores' in the search dropdown")
    print(f"[top_scores_proxy] press Ctrl+C to stop")

    server = HTTPServer(("localhost", _PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[top_scores_proxy] stopped.")


if __name__ == "__main__":
    main()
