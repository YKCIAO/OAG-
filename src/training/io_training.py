from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez(path, **arrays)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def try_export_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """
    Optional: export using pandas if available. If pandas isn't available, saves jsonl instead.
    """
    ensure_dir(os.path.dirname(path))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
    except Exception:
        # fallback: jsonl
        jsonl_path = os.path.splitext(path)[0] + ".jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
