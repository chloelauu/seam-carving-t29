"""Regenerate assets/evidence/source_hashes.json from the files it covers.

Usage (from the repository root):

    python -m scripts.hashes          # rewrite the manifest
    python -m scripts.hashes --check  # exit 1 if the manifest is stale

The manifest is a hash of the code as committed, one entry per file: it says only that these are
these files, at this revision. It cannot show that a measurement predates a code change — the tests
job fails when a covered file no longer matches, and a person then regenerates the manifest. Keep
the evidence and the manifest in the same commit when the code behind the evidence changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "assets" / "evidence" / "source_hashes.json"

# The code that produces the evidence in assets/evidence/: the package, the app, the scripts
# that generate it (including the graph-cut last-column counts the report cites), and the tests
# that pin its behaviour. The workflow is deliberately not listed — it runs the code rather than
# producing evidence, and hashing it would make every workflow edit invalidate the manifest.
COVERED = [
    "app.py",
    "pyproject.toml",
    "scripts/benchmark.py",
    "scripts/ci_figure.py",
    "scripts/create_example.py",
    "scripts/evaluate.py",
    "scripts/hashes.py",
    "scripts/last_column.py",
    "seamcarve/core.py",
    "seamcarve/__init__.py",
    "tests/test_app.py",
    "tests/test_regressions.py",
    "tests/test_seamcarve.py",
]


def digest(path: pathlib.Path) -> str:
    # Hash the content as the repository stores it, not as the working tree presents it: a
    # Windows clone with core.autocrlf hands out CRLF, and hashing those bytes would fail
    # --check on a file nobody touched, then fail CI if the result were committed.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build() -> dict[str, str]:
    missing = [p for p in COVERED if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit("cannot hash what is not there: " + ", ".join(missing))
    return {p: digest(ROOT / p) for p in COVERED}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the committed manifest still matches, without writing",
    )
    args = parser.parse_args()

    fresh = build()
    if args.check:
        if not MANIFEST.is_file():
            print(f"{MANIFEST.relative_to(ROOT)} is missing", file=sys.stderr)
            return 1
        recorded = json.loads(MANIFEST.read_text())
        stale = {
            path: (recorded.get(path), actual)
            for path, actual in fresh.items()
            if recorded.get(path) != actual
        }
        extra = sorted(set(recorded) - set(fresh))
        for path, (was, now) in stale.items():
            print(f"stale: {path}\n  recorded {was}\n  actual   {now}", file=sys.stderr)
        if extra:
            print("no longer covered: " + ", ".join(extra), file=sys.stderr)
        if stale or extra:
            print("run `python -m scripts.hashes` to regenerate", file=sys.stderr)
            return 1
        print(f"manifest matches all {len(fresh)} files")
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(fresh, indent=2) + "\n", newline="\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(fresh)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
