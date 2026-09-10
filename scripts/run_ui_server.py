#!/usr/bin/env python3
"""
Launcher script for the HypoTrader Range Sweep V2 Web UI & Execution Map Server.
Usage:
    python3 scripts/run_ui_server.py [port]
"""

import sys
import os

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.web.server import start_server, start_server as run_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    run_server(port=port)
