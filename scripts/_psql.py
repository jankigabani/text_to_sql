"""Find psql.exe on Windows even when it is not on PATH.

PostgreSQL's Windows installer does not add its bin folder to PATH, and it can be installed
on any drive (this project's machine has it on D:). Looked for, in order:

1. the --psql argument or the PSQL environment variable;
2. `psql` on PATH;
3. <drive>:\\**\\PostgreSQL\\<version>\\bin\\psql.exe on C:, D:, E: (newest version first);
4. the folder of the running PostgreSQL Windows service.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PATTERNS = [
    r"{d}:\PostgreSQL\*\bin\psql.exe",
    r"{d}:\Program Files\PostgreSQL\*\bin\psql.exe",
    r"{d}:\Program Files (x86)\PostgreSQL\*\bin\psql.exe",
    r"{d}:\*\PostgreSQL\*\bin\psql.exe",
    r"{d}:\*\*\PostgreSQL\*\bin\psql.exe",
]


def _version_key(p: str) -> float:
    m = re.search(r"PostgreSQL[\\/](\d+(?:\.\d+)?)", p, re.I)
    return float(m.group(1)) if m else 0.0


def _from_service() -> str | None:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Service -Filter \"Name like 'postgresql%'\").PathName"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        m = re.search(r'"?([A-Za-z]:\\[^"]+?\\bin)\\pg_ctl\.exe', line, re.I)
        if m:
            cand = Path(m.group(1)) / "psql.exe"
            if cand.exists():
                return str(cand)
    return None


def find_psql(explicit: str | None = None) -> str | None:
    for cand in (explicit, os.environ.get("PSQL")):
        if cand and (shutil.which(cand) or Path(cand).exists()):
            return shutil.which(cand) or cand
    on_path = shutil.which("psql")
    if on_path:
        return on_path
    if sys.platform != "win32":
        return None
    found: list[str] = []
    for d in "CDE":
        if not Path(f"{d}:\\").exists():
            continue
        for pat in PATTERNS:
            found += glob.glob(pat.format(d=d))
    if found:
        return sorted(set(found), key=_version_key, reverse=True)[0]
    return _from_service()


if __name__ == "__main__":
    print(find_psql() or "psql not found")
