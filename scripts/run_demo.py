"""
scripts/run_demo.py — Lightweight HTTP server for SWE Agent interactive demo UI.
Serves demo/index.html on http://localhost:3000 (or specified port).
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import sys
from pathlib import Path


def run_demo(port: int = 3000) -> None:
    demo_dir = Path(__file__).resolve().parent.parent / "demo"
    if not demo_dir.exists() or not (demo_dir / "index.html").exists():
        print(f"Error: Demo directory or index.html not found at {demo_dir}")
        sys.exit(1)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(demo_dir))

    # Allow socket address reuse
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"🚀 SWE Agent Demo UI is running at http://localhost:{port}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down demo server.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    run_demo(port)
