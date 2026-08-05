#!/usr/bin/env python3
"""Small Python frontend for the Rust SCF_Szabo executable.

The numerical implementation now lives in Rust (`src/main.rs`). This script keeps a
Python entry point for learners who prefer Python tooling: it builds/runs the Rust
binary through Cargo and streams the calculation output.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_frontend(release: bool) -> int:
    """Build and run the Rust SCF program with Cargo."""
    if shutil.which("cargo") is None:
        print("error: cargo was not found; install Rust from https://rustup.rs/", file=sys.stderr)
        return 127

    command = ["cargo", "run"]
    if release:
        command.append("--release")

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Python frontend for the Rust SCF_Szabo calculation")
    parser.add_argument("--release", action="store_true", help="run the optimized Rust binary with cargo run --release")
    args = parser.parse_args()
    return run_frontend(args.release)


if __name__ == "__main__":
    raise SystemExit(main())
