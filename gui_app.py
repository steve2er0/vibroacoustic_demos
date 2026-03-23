#!/usr/bin/env python3
"""
Force reconstruction — entry point.

**Default (macOS, Windows, Linux):** Streamlit single-page GUI
(``gui_streamlit``): ``python3 gui_app.py``

**Force web GUI explicitly:** ``python3 gui_app.py --web``

**Legacy Matplotlib GUI:** ``FORCE_RECON_GUI=mpl python3 gui_app.py``
**Legacy full-window Tk UI:** ``FORCE_RECON_GUI=tk python3 gui_app.py``

**CLI / headless** (paths + optional ``--save``, no interactive slice):
``python3 gui_app.py --fx … --fy … --fz … --flight … --t0 0 --t1 1 --save out.csv``
or
``python3 gui_app.py --fx … --fy … --fz … --flight-mat ch01.mat ch02.mat ch03.mat --t0 0 --t1 1``
or
``python3 gui_app.py --ones-h --flight-mat ch01.mat ch02.mat ch03.mat --t0 0 --t1 1``

``python3 gui_app.py --help`` — full CLI flags (delegated).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    force_gui = os.environ.get("FORCE_RECON_GUI", "").lower()
    explicit_web_arg = len(sys.argv) > 1 and sys.argv[1].lower() in {"--web", "--streamlit"}
    default_to_web = len(sys.argv) == 1 and force_gui not in {"tk", "mpl", "matplotlib"}
    want_web = force_gui in {"web", "streamlit"} or explicit_web_arg or default_to_web
    if want_web:
        app_path = Path(__file__).with_name("gui_streamlit.py")
        if importlib.util.find_spec("streamlit") is None:
            if force_gui in {"web", "streamlit"} or explicit_web_arg:
                print(
                    "Streamlit not installed. Run: python3 -m pip install streamlit",
                    file=sys.stderr,
                )
                return 1
            print("Streamlit not installed; falling back to Matplotlib GUI.", file=sys.stderr)
        else:
            cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
            if len(sys.argv) > 2:
                cmd.extend(sys.argv[2:])
            return subprocess.call(cmd)

    if force_gui == "tk":
        from gui_tk import main as tk_main

        tk_main()
        return 0

    # Any CLI flag → headless / argparse path (same as former gui_macos)
    if len(sys.argv) > 1:
        from gui_macos import main as cli_main

        return cli_main()

    from gui_interactive import run_gui

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
