#!/usr/bin/env python3
"""Track 4 entrypoint wrapper delegating to `tracks/track4/jobs/manage.py`."""

from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "jobs" / "manage.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
