"""Native file open/save dialogs (no Matplotlib). macOS: AppleScript; else: Tk dialog."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def choose_open_file(
    title: str,
    *,
    filetypes=None,
) -> Path | None:
    filetypes = filetypes or [("CSV", "*.csv"), ("All", "*.*")]
    if sys.platform == "darwin":
        safe = title.replace('"', "'")
        script = f'POSIX path of (choose file with prompt "{safe}")'
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return None
        p = r.stdout.strip()
        return Path(p) if p else None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        p = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
        )
    finally:
        root.destroy()
    return Path(p) if p else None


def choose_open_files(
    title: str,
    *,
    filetypes=None,
) -> list[Path]:
    filetypes = filetypes or [("CSV / MAT", ("*.csv", "*.mat")), ("All", "*.*")]
    if sys.platform == "darwin":
        safe = title.replace('"', "'")
        script = (
            f'set xs to choose file with prompt "{safe}" with multiple selections allowed true\n'
            'set out to ""\n'
            'repeat with f in xs\n'
            'set out to out & POSIX path of f & linefeed\n'
            'end repeat\n'
            'return out'
        )
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return []
        return [Path(line) for line in r.stdout.splitlines() if line.strip()]

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        paths = filedialog.askopenfilenames(
            title=title,
            filetypes=filetypes,
        )
    finally:
        root.destroy()
    return [Path(p) for p in paths if p]


def choose_save_file(default_name: str) -> Path | None:
    if sys.platform == "darwin":
        dn = default_name.replace('"', "'")
        script = f'POSIX path of (choose file name default name "{dn}")'
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return None
        p = r.stdout.strip()
        return Path(p) if p else None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        p = filedialog.asksaveasfilename(
            title="Save F̂ spectrum",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=default_name,
        )
    finally:
        root.destroy()
    return Path(p) if p else None
