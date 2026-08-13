#!/usr/bin/env python3
"""Entry point. Run from the folder holding the frames, or pass --source.

    python run.py scan
    python run.py all --source /path/to/photos
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meteor.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
