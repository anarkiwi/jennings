"""Check that tests fail when the code they cover is broken on purpose.

    python tools/break_check.py SPEC [--only NAME]... [--command COMMAND]

``SPEC`` (TOML): a shell ``command`` that passes as the code is, and ``[[break]]`` tables of
``name``, ``file``, text ``old`` occurring once in it and ``new`` put in its place while the
command (or the break's own ``command``) runs. Exits 1 unless every break makes it fail.
"""

import argparse
import pathlib
import subprocess
import sys
import tomllib

from tqdm import tqdm


def check(brk, command):
    """Why ``brk`` is not caught by ``command``, or None: its file is put back either way."""
    path = pathlib.Path(brk["file"])
    text = path.read_text(encoding="utf-8")
    if text.count(brk["old"]) != 1:
        return f"its text occurs {text.count(brk['old'])} times in {path}"
    try:
        path.write_text(text.replace(brk["old"], brk["new"]), encoding="utf-8")
        ran = subprocess.run(command, shell=True, capture_output=True, check=False)
    finally:
        path.write_text(text, encoding="utf-8")
    return None if ran.returncode else f"`{command}` still passes"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("spec")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--command")
    args = ap.parse_args(argv)
    with open(args.spec, "rb") as f:
        spec = tomllib.load(f)
    breaks = [b for b in spec["break"] if not args.only or b["name"] in args.only]
    missed = 0
    for brk in tqdm(breaks, unit="break", disable=None):
        why = check(brk, args.command or brk.get("command", spec["command"]))
        missed += why is not None
        tqdm.write(f"{brk['name']}: {why or 'caught'}")
        sys.stdout.flush()
    print(f"{len(breaks) - missed} of {len(breaks)} breaks caught", flush=True)
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
