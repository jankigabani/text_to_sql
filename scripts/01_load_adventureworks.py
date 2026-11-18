"""Load the AdventureWorks OLTP database into PostgreSQL — 68 tables, 180+ foreign keys.

Why this database: it is a real, normalised, multi-schema OLTP design (sales, production,
person, purchasing, humanresources) with the join complexity the research question is
about, and it is public, so the experiment is reproducible.

What this script does:

1. downloads Microsoft's "AdventureWorks 2014 OLTP Script" (CSV data) — ~30 MB;
2. downloads Lorin Thwaits' PostgreSQL port of the schema (install.sql);
3. converts the CSVs to the form install.sql expects (this is a Python port of the
   project's update_csvs.rb, so no Ruby install is needed);
4. creates the database and runs install.sql via psql.

    python scripts/01_load_adventureworks.py                 # full run
    python scripts/01_load_adventureworks.py --skip-download  # reuse data/adventureworks
    python scripts/01_load_adventureworks.py --database awtest

Needs `psql`. It does not have to be on PATH: the script finds it under
<drive>:\\...\\PostgreSQL\\<version>\\bin on Windows (or pass --psql).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings   
from scripts._psql import find_psql   

OLTP_ZIP = (
    "https://github.com/Microsoft/sql-server-samples/releases/download/"
    "adventureworks/AdventureWorks-oltp-install-script.zip"
)
INSTALL_SQL = (
    "https://raw.githubusercontent.com/lorint/AdventureWorks-for-Postgres/"
    "master/install.sql"
)


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def convert_csv(path: Path) -> bool:
    """Port of update_csvs.rb. Returns True if the file was rewritten.

    Microsoft ships these CSVs with '+|' as the column separator, '&|' as the row
    terminator, a UTF-8 BOM, embedded NULs in hierarchyid columns, and raw binary for
    the GIF/geography columns. PostgreSQL's COPY needs tab-separated text with none of that.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-16", errors="replace")

    had_bom = text.startswith("\ufeff")
    if had_bom:
        text = text[1:]

    is_pipes = "+|" in text[:4000]
    needs_clean = is_pipes or "&|" in text[:4000] or "\tE6100000010C" in text[:4000]
    if not (needs_clean or had_bom):
        return False

    if is_pipes:
        text = text.replace("|474946383961", "|\\x474946383961").replace('"', '""')
        out_rows: list[str] = []
        buffer = ""
        for line in text.splitlines(keepends=True):
            while "&|" in line:
                head, _, line = line.partition("&|")
                buffer += head.strip()
                fields = []
                for i, part in enumerate(buffer.split("+|")):
                    if i == 0 and part == "\x00":
                        part = "00"
                    else:
                        part = part.replace("\x00", "")
                    if (part[:1] == "<" and part[-1:] == ">") or \
                       (part[1:2] == "<" and part[-1:] == ">"):
                        part = '"' + part.lstrip("\ufeff") + '"'
                    elif "\t" in part:
                        part = '"' + part + '"'
                    fields.append(part)
                out_rows.append("\t".join(fields))
                buffer = ""
            if line.strip():
                buffer += line.replace("\r\n", "\\n").replace("\n", "\\n")
        if buffer.strip():
            out_rows.append(buffer)
        new_text = "\n".join(out_rows) + "\n"
    else:
        new_text = (
            text.replace('"', '""')
            .replace("&|\r\n", "\n")
            .replace("&|\n", "\n")
            .replace("\tE6100000010C", "\t\\xE6100000010C")
            .replace("\r\n", "\n")
        )

    path.write_text(new_text, encoding="utf-8", newline="")
    return True


def run_psql(psql: str, args: list[str], env: dict, cwd: Path | None = None,
             stdin_file: Path | None = None, check: bool = True) -> str:
    cmd = [psql, "-v", "ON_ERROR_STOP=0", *args]
    # utf-8 + replace: Windows consoles default to cp1252, and the AdventureWorks data
    # has accented names, which would otherwise crash the decoding of psql's output.
    kw = dict(env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
              errors="replace")
    if stdin_file is not None:
        with stdin_file.open("rb") as f:
            proc = subprocess.run(cmd, stdin=f, **kw)
    else:
        proc = subprocess.run(cmd, **kw)
    errors = [ln for ln in (proc.stderr or "").splitlines() if "ERROR" in ln or "FATAL" in ln]
    for line in (errors or (proc.stderr or "").strip().splitlines())[-8:]:
        print(f"    psql: {line}")
    if proc.returncode != 0 and stdin_file is None and check:
        raise SystemExit(f"psql failed: {' '.join(cmd)}")
    return proc.stdout or ""


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="data/adventureworks")
    ap.add_argument("--database", default=s.pgdatabase)
    ap.add_argument("--psql", default=None, help="path to psql(.exe); found automatically")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--keep-existing-db", action="store_true")
    a = ap.parse_args()

    psql = find_psql(a.psql)
    if not psql:
        print("[FAIL] psql not found (not on PATH, and not under <drive>:\\...\\PostgreSQL\\<v>\\bin).")
        print("       Run:  python scripts\\01_load_adventureworks.py "
              '--psql "D:\\Janki\\PostgreSQL\\18\\bin\\psql.exe"')
        return 1
    print(f"using psql: {psql}")
    a.psql = psql

    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    print("\n1/4 downloading sources")
    if not a.skip_download:
        zip_path = download(OLTP_ZIP, work / "aw-oltp.zip")
        print("  extracting CSVs ...")
        with zipfile.ZipFile(zip_path) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".csv")]
            for m in members:
                target = work / Path(m).name
                with z.open(m) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        print(f"  extracted {len(members)} CSV files")
    download(INSTALL_SQL, work / "install.sql")

    print("\n2/4 converting CSVs for PostgreSQL")
    csvs = sorted(work.glob("*.csv"))
    if not csvs:
        print("[FAIL] no CSV files found — rerun without --skip-download")
        return 1
    changed = sum(1 for c in csvs if convert_csv(c))
    print(f"  rewrote {changed} of {len(csvs)} files")

    print(f"\n3/4 creating database '{a.database}'")
    env = {**os.environ, "PGPASSWORD": s.pgpassword, "PGCLIENTENCODING": "UTF8"}
    base = ["-h", s.pghost, "-p", str(s.pgport), "-U", s.pguser]
    if not a.keep_existing_db:
        # WITH (FORCE) closes other sessions first (pgAdmin keeps one open); PG 13+.
        try:
            run_psql(a.psql, [*base, "-d", "postgres", "-c",
                              f'DROP DATABASE IF EXISTS "{a.database}" WITH (FORCE)'], env)
        except SystemExit:
            run_psql(a.psql, [*base, "-d", "postgres", "-c",
                              f'DROP DATABASE IF EXISTS "{a.database}"'], env)
    run_psql(a.psql, [*base, "-d", "postgres", "-c",
                      f'CREATE DATABASE "{a.database}" ENCODING \'UTF8\' TEMPLATE template0'], env)

    print("\n4/4 running install.sql (this takes a minute)")
    # install.sql uses \copy with relative paths, so psql must run inside the CSV folder.
    run_psql(a.psql, [*base, "-d", a.database], env, cwd=work,
             stdin_file=work / "install.sql")

    print("\nverifying")
    out = run_psql(a.psql, [*base, "-d", a.database, "-tA", "-c",
                            "SELECT count(*) FROM pg_class c JOIN pg_namespace n "
                            "ON n.oid=c.relnamespace WHERE c.relkind='r' AND n.nspname NOT IN "
                            "('pg_catalog','information_schema')"], env)
    n_tables = int(out.strip() or 0)
    print(f"  tables loaded: {n_tables}  (expect about 68)")
    run_psql(a.psql, [*base, "-d", a.database, "-c", "ANALYZE"], env)
    if n_tables < 60:
        print("[FAIL] fewer tables than expected — read the psql errors above.")
        return 1

    print(f"\nDone. Set PGDATABASE={a.database} in .env, then run:")
    print("  python scripts/02_amplify.py --target-rows 10000000")
    print("  python scripts/03_create_readonly_role.py")
    print("  python scripts/04_build_schema_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
