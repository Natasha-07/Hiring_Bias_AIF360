"""Configure imports so local ./AIF360 source is used when available."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_aif360_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    local_aif360_root = repo_root / "AIF360"

    if not local_aif360_root.is_dir():
        return

    local_aif360_root_str = str(local_aif360_root)
    if local_aif360_root_str not in sys.path:
        # Put local source first so imports resolve without pip installing aif360.
        sys.path.insert(0, local_aif360_root_str)


_ensure_local_aif360_on_path()
