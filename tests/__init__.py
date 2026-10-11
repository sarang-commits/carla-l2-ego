"""Test package.

Ensure the project root (parent of this package) is importable so that
``telemetry``, ``sensors`` and ``tests.fakes`` resolve regardless of how the
test runner sets its top-level directory.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
