"""Thin shim — calls ``mtg_bot.cli.shard.main()`` so existing SSH
invocations keep working.

The real implementation lives at ``src/mtg_bot/cli/shard.py``;
``distributed.py`` builds remote SSH commands that invoke this path
verbatim, and the bundle/queue server hands out a snapshot of ``main``
that includes this file. Don't add logic here — edit the cli module.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When invoked over SSH with ``PYTHONPATH=src`` already set, the import
# below works directly. When invoked with bare python (e.g. an old
# operator habit), prepend ``src/`` so the editable install still
# resolves.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mtg_bot.cli.shard import main  # noqa: E402

if __name__ == "__main__":
    main()
