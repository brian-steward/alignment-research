import os
import sys

# Make the flat modules importable as top-level (e.g. `from schema import Config`).
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
