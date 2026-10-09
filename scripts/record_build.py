#!/usr/bin/env python3
"""Record the successful Makefile compiler invocation alongside its executable."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    version = subprocess.run([command[0], "--version"], check=True, capture_output=True, text=True)
    files = [ROOT / "Makefile"] + [p for directory in ("src", "include")
                                 for p in sorted((ROOT / directory).glob("*")) if p.is_file()]
    receipt = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": command,
               "compiler_version": version.stdout, "binary_sha256": sha256(args.binary),
               "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files}}
    args.binary.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
