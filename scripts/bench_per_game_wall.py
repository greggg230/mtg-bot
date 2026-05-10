"""Thin shim — calls ``mtg_bot.cli.bench_kinds.per_game_wall.main()``
so existing operator habits (``python scripts/bench_per_game_wall.py
...``) keep working.

The real implementation lives at
``src/mtg_bot/cli/bench_kinds/per_game_wall.py``. Don't add logic
here — edit the cli module. New invocations should prefer:
``python -m mtg_bot.cli bench per-game-wall ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mtg_bot.cli.bench_kinds.per_game_wall import main  # noqa: E402

if __name__ == "__main__":
    main()
