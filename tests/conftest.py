"""Pytest configuration: make project modules importable from the repo root.

This lets `pytest` run from anywhere while tests `import main`, `import
analysis`, etc., and resolve the bundled config.yaml / sample CSV via the
project root.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Run tests with the project root as CWD so relative paths in config.yaml
# (e.g. the default sample portfolio) resolve.
os.chdir(ROOT)
