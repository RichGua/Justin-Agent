from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import AgentConfig
from .runtime import JustinRuntime, build_runtime_bundle
from .types import to_plain_dict


def build_server(runtime: JustinRuntime, host: str, port: int) -> ThreadingHTTPServer:
    web_dir = Path(__file__).with_name("web")

    class AgentRequestHandler(BaseHTTPRequestHandler):
        server_version = "Justin/0.1"

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/api/health":
                    return self._json({"ok": True})
                if path == "/api/sessions":
                    return self._json({"items": [to_plain_dict(item) for item in runtime.list_sessions()]})
                if path.startswith("/api/sessions/"):
                    session_id = path.split("/")[-1]
                    return self._json(
                        {
                            "session_id": session_id,
                            "messages": [to_plain_dict(item) for item in runtime.list_messages(session_id)],
                        }
                    )
                if path == "/api/candidates":
                    status = parse_qs(parsed.query).get("status", [None])[0]
                    return self._json({"items": [to_plain_dict(item) for item in runtime.list_candidates(status=status)]})
                if path == "/api/memories":
                    kind = parse_qs(parsed.query).get("kind", [None])[0]
                    return self._json({"items": [to_plain_dict(item) for item in runtime.list_memories(kind=kind)]})
                if path == "/api/search":
                    query = parse_qs(parsed.query).get("q", [""])[0]
                    return self._json({"items": [to_plain_dict(item) for item in runtime.search_memories(query)]})
                if path == "/api/state":
                    return self._json(
                        {
                            "sessions": [to_plain_dict(item) for item in runtime.list_sessions()],
                            "candidates": [to_plain_dict(item) for item in runtime.list_candidates(status="pending")],
                            "memories": [to_plain_dict(item) for item in runtime.list_memories()],
                        }
                    )

                return self._serve_static(path, web_dir)
            except Exception as exc:  # pragma: no cover - last-resort HTTP guard
                return self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                payload = self._read_json()

                if path == "/api/messages":
                    content = str(payload.get("content", "")).strip()
                    session_id = payload.get("session_id")
                    if not content:
                        return self._json({"error": "content is required"}, HTTPStatus.BAD_REQUEST)
                    result = runtime.send_message(content=content, session_id=session_id)
                    return self._json(to_plain_dict(result))

                if path == "/api/sessions":
                    title = str(payload.get("title", "New session")).strip() or "New session"
                    session = runtime.store.create_session(title)
                    return self._json(to_plain_dict(session), HTTPStatus.CREATED)

                if path.startswith("/api/candidates/") and path.endswith("/confirm"):
                    candidate_id = path.split("/")[-2]
                    try:
                        memory = runtime.confirm_candidate(candidate_id)
                    except KeyError as exc:
                        return self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                    return self._json({"memory": to_plain_dict(memory)})

                if path.startswith("/api/candidates/") and path.endswith("/reject"):
                    candidate_id = path.split("/")[-2]
                    note = payload.get("note")
                    try:
                        candidate = runtime.reject_candidate(candidate_id, note)
                    except KeyError as exc:
                        return self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                    return self._json({"candidate": to_plain_dict(candidate)})

                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - last-resort HTTP guard
                return self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

        def _read_json(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, path: str, web_dir: Path) -> None:
            if path in {"/", ""}:
                file_path = web_dir / "index.html"
            else:
                file_path = web_dir / path.lstrip("/")

            if not file_path.exists() or not file_path.is_file():
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            body = file_path.read_bytes()
            media_type = "text/plain; charset=utf-8"
            if file_path.suffix == ".html":
                media_type = "text/html; charset=utf-8"
            elif file_path.suffix == ".css":
                media_type = "text/css; charset=utf-8"
            elif file_path.suffix == ".js":
                media_type = "application/javascript; charset=utf-8"

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), AgentRequestHandler)


def serve(config: AgentConfig | None = None) -> None:
    bundle = build_runtime_bundle(config)
    runtime = JustinRuntime(bundle)
    server = build_server(runtime, bundle.config.host, bundle.config.port)
    print(f"Justin listening on http://{bundle.config.host}:{bundle.config.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
