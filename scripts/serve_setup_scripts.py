"""Thin shim — calls ``mtg_bot.cli.serve.main()`` so existing user
workflows (``python scripts/serve_setup_scripts.py``) keep working.

The real implementation lives at ``src/mtg_bot/cli/serve.py``.
Don't add logic here — edit the cli module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mtg_bot.cli.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
