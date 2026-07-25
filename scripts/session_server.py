#!/usr/bin/env python3
"""Localhost session-logging server: the dashboard, but writable.

    python scripts/session_server.py          # then open http://127.0.0.1:8765

Serves the same self-contained dashboard as scripts/export_session_viewer.py with
the in-page "Log a session" form enabled. The form's whole point is minimal typing:
the server scans data/*.txt (and data/uploads/*.txt) hand-history exports, clusters
hands into sessions by time gap (sessions.detect_sessions), drops the windows already
in the ledger, and proposes the rest with times/hands/stakes/table-result prefilled --
leaving only the client-UI-only numbers (rakeback balance, rake-paid counter,
optionally main balances) to type. See src/sessions.py for which fields are which.

Plain stdlib http.server, bound to 127.0.0.1 only (personal-machine tool: no auth,
no TLS, and it must never be bound to a public interface). Endpoints:

    GET  /            the dashboard (payload rebuilt per request)
    GET  /api/detect  unlogged-session proposals from the export scan
    POST /api/log     JSON session fields -> validate -> append to the CSV
    POST /api/upload  raw .txt export body (X-Filename header) -> data/uploads/
"""
from __future__ import annotations

import argparse
import json
import re
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hand_history
import session_analysis
import sessions
from export_session_viewer import build_payload, build_population_payload, render_html

DEFAULT_PORT = 8765
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # a 5-month export is ~13k hands / a few MB


class SessionServer(ThreadingHTTPServer):
    """Holds the paths/config; handlers reach it via self.server."""

    def __init__(
        self,
        addr,
        csv_path: str,
        data_dir: str,
        currency: str = "₮",
        grading: bool = True,
        grading_ctx: session_analysis.GradingContext | None = None,
        cache_dir: str = "cache/",
    ):
        super().__init__(addr, _Handler)
        self.csv_path = csv_path
        self.data_dir = Path(data_dir)
        self.currency = currency
        self.grading = grading
        self.cache_dir = cache_dir
        self._hands_cache: dict[Path, tuple[float, list]] = {}  # path -> (mtime, hands)
        self._graded_cache: dict[Path, tuple[float, list]] = {}  # path -> (mtime, GradedHand rows)
        self._grading_ctx = grading_ctx
        self._grading_ctx_loaded = grading_ctx is not None
        self._pop_ranges = None  # 4-max RangeMap for the population card (lazy, load-only)
        self._pop_ranges_loaded = False

    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn(), a reverse-DNS lookup that can
        # hang ~30s on macOS for loopback binds; the FQDN is never used here, so bind
        # at the TCPServer level and fill the fields from the literal address instead.
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]

    def _export_paths(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.txt")) + sorted((self.data_dir / "uploads").glob("*.txt"))

    def scan_hands(self) -> list:
        """All hands across every export under data_dir (top level + uploads/),
        parsed once per file mtime -- rescanning between requests is then free."""
        all_hands: list = []
        for path in self._export_paths():
            mtime = path.stat().st_mtime
            cached = self._hands_cache.get(path)
            if cached is None or cached[0] != mtime:
                hands, _failures = hand_history.parse_file(str(path))
                self._hands_cache[path] = (mtime, hands)
            all_hands.extend(self._hands_cache[path][1])
        return all_hands

    def grading_context(self) -> session_analysis.GradingContext | None:
        """Injected context (tests), else built lazily once from cache_dir's solver
        artifacts; None when grading is off or the equity table isn't on disk."""
        if not self.grading:
            return None
        if not self._grading_ctx_loaded:
            self._grading_ctx = session_analysis.make_grading_context(self.cache_dir)
            self._grading_ctx_loaded = True
        return self._grading_ctx

    def fourmax_ranges(self) -> dict | None:
        """Solved 4-max RangeMap for the population card's GTO baselines: reuse the
        grading context's copy when grading already loaded one, else one lazy
        load-only read of the pickle; None when it isn't on disk (card degrades)."""
        ctx = self.grading_context()
        if ctx is not None and ctx.fourmax is not None:
            return ctx.fourmax[1]
        if not self._pop_ranges_loaded:
            loaded = session_analysis.load_fourmax_ranges_if_cached(self.cache_dir)
            self._pop_ranges = loaded[1] if loaded is not None else None
            self._pop_ranges_loaded = True
        return self._pop_ranges

    def scan_graded(self) -> list | None:
        """GradedHand rows across every export, same per-file-mtime pattern as
        scan_hands with a persisted JSON layer underneath (session_grades/) so a
        server restart never regrades unchanged files. None = grading unavailable."""
        ctx = self.grading_context()
        if ctx is None:
            return None
        self.scan_hands()  # refresh _hands_cache so each file's parse is reused below
        all_rows: list = []
        for path in self._export_paths():
            mtime = path.stat().st_mtime
            cached = self._graded_cache.get(path)
            if cached is None or cached[0] != mtime:
                rows = session_analysis.load_or_grade_file(
                    str(path), self._hands_cache[path][1], ctx, cache_dir=self.cache_dir)
                self._graded_cache[path] = (mtime, rows)
            all_rows.extend(self._graded_cache[path][1])
        return all_rows


def _stakes_label(hands) -> str:
    return " + ".join(f"{bb:g}bb" for bb in sorted({h.bb_amount for h in hands}))


def _proposals(server: SessionServer) -> list[dict]:
    """Detected sessions not overlapping any ledger row, ready for the form."""
    ledger, _failures = sessions.load_sessions(server.csv_path)
    intervals = [
        (hand_history.parse_timestamp(s.start_time), hand_history.parse_timestamp(s.end_time))
        for s in ledger
    ]
    proposals = []
    for detected in sessions.detect_sessions(server.scan_hands()):
        start = hand_history.parse_timestamp(detected.start_time)
        end = hand_history.parse_timestamp(detected.end_time)
        if any(start <= b and end >= a for a, b in intervals):
            continue  # already logged (any overlap counts -- windows never split rows)
        nets = [sessions.hero_net_currency(h) for h in detected.hands]
        proposals.append({
            "start": detected.start_time,
            "end": detected.end_time,
            "hands": len(detected.hands),
            "stakes": _stakes_label(detected.hands),
            "hhNet": round(sum(n for n in nets if n is not None), 2),
            "excluded": sum(1 for n in nets if n is None),
        })
    return proposals


class _Handler(BaseHTTPRequestHandler):
    server: SessionServer

    def log_message(self, format, *args):  # noqa: A002 -- stdlib signature
        pass  # keep the terminal quiet; errors surface as JSON responses instead

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"body too large ({length} bytes)")
        return self.rfile.read(length)

    def do_GET(self) -> None:  # noqa: N802 -- stdlib naming
        if self.path == "/" or self.path == "/index.html":
            hands = self.server.scan_hands()
            payload = build_payload(
                self.server.csv_path,
                hands=hands,
                currency=self.server.currency,
                served=True,
                graded=self.server.scan_graded(),
                population=build_population_payload(hands, self.server.fourmax_ranges()),
            )
            self._send(200, render_html(payload).encode(), "text/html; charset=utf-8")
        elif self.path == "/api/detect":
            self._send_json(200, {"proposals": _proposals(self.server)})
        else:
            self._send_json(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 -- stdlib naming
        try:
            if self.path == "/api/log":
                self._handle_log()
            elif self.path == "/api/upload":
                self._handle_upload()
            else:
                self._send_json(404, {"error": f"no route {self.path}"})
        except Exception as exc:  # noqa: BLE001 -- every failure becomes a form message
            self._send_json(400, {"error": str(exc)})

    def _handle_log(self) -> None:
        body = json.loads(self._read_body())
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object of session fields")
        # _session_from_row accepts CSV strings and JSON scalars alike, and Session's
        # own __post_init__ carries the real validation -- errors come back verbatim.
        session = sessions._session_from_row(body)
        sessions.append_session(self.server.csv_path, session)
        self._send_json(200, {"ok": True})

    def _handle_upload(self) -> None:
        raw_name = Path(unquote(self.headers.get("X-Filename", "export.txt"))).name
        safe_stem = re.sub(r"[^\w.-]", "_", Path(raw_name).stem) or "export"
        text = self._read_body().decode("utf-8", errors="replace")
        if not hand_history.split_hands(text):
            raise ValueError("no CoinPoker hands found in the uploaded file")
        uploads = self.server.data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        out = uploads / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_stem}.txt"
        out.write_text(text, encoding="utf-8")
        self._send_json(200, {"ok": True, "saved": out.name})


def make_server(addr, csv_path: str, data_dir: str, currency: str = "₮",
                grading: bool = True, grading_ctx=None, cache_dir: str = "cache/") -> SessionServer:
    return SessionServer(addr, csv_path=csv_path, data_dir=data_dir, currency=currency,
                         grading=grading, grading_ctx=grading_ctx, cache_dir=cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/sessions.csv")
    parser.add_argument("--data-dir", default="data/",
                        help="directory whose *.txt exports are scanned for sessions")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--currency", default="₮")
    parser.add_argument("--cache-dir", default="cache/",
                        help="solver artifacts (equity table, 4-max ranges) + grade cache")
    parser.add_argument("--no-grade", action="store_true",
                        help="skip the per-session GTO grade & luck cards")
    args = parser.parse_args()

    server = make_server(("127.0.0.1", args.port), csv_path=args.csv,
                         data_dir=args.data_dir, currency=args.currency,
                         grading=not args.no_grade, cache_dir=args.cache_dir)
    if not args.no_grade:
        # Warm once before serving so page loads never pay first-solve latency.
        print("warming grade cache (first ever run may take a minute)...")
        rows = server.scan_graded()
        if rows is None:
            print(f"grading unavailable: no equity table under {args.cache_dir} "
                  "(see README's backtesting section)")
        else:
            print(f"grade cache ready ({len(rows)} graded spots)")
    print(f"session tracker: http://127.0.0.1:{server.server_address[1]}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
