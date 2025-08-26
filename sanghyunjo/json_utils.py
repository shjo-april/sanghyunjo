# Copyright (C) 2025 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import numpy as np
import json as _json

from typing import Any, List, Iterable

try:
    import orjson as _orjson
    _HAVE_ORJSON = True
except Exception:
    _orjson = None
    _HAVE_ORJSON = False
    print("orjson is not installed. If you want to use orjson as backend, please use \"pip install orjson\".")

def jsread(filepath: str, backend: str = "json") -> Any:
    """
    Read JSON file with selectable backend.
    """
    if backend == "orjson" and _HAVE_ORJSON:
        with open(filepath, "rb") as f:
            data = f.read()
        return _orjson.loads(data)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return _json.load(f)

def jswrite(filepath: str, data: Any, backend: str = "json") -> None:
    """
    Write JSON file with selectable backend.
    """
    if backend == "orjson" and _HAVE_ORJSON:
        opt = 0
        opt |= _orjson.OPT_APPEND_NEWLINE  # type: ignore
        opt |= _orjson.OPT_INDENT_2  # force 2-space when using orjson

        try:
            blob = _orjson.dumps(data, option=opt)  # type: ignore
        except TypeError:
            # auto-enable numpy support on demand
            opt |= _orjson.OPT_SERIALIZE_NUMPY  # type: ignore
            blob = _orjson.dumps(data, option=opt)  # type: ignore
        
        with open(filepath, "wb") as f:
            f.write(blob)
    else:
        # std json: keep '\t' as tabs; auto-convert numpy via default
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            _json.dump(data, f, ensure_ascii=False, indent='\t')

# ---------------------------
# NDJSON (JSON Lines) helpers
# ---------------------------
def jslread(filepath: str, backend: str = "json") -> List[Any]:
    """
    Read NDJSON (one JSON object per line). Empty/whitespace lines are skipped.
    Returns a list of parsed rows.
    """
    rows: List[Any] = []

    if backend == "orjson":
        with open(filepath, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(_orjson.loads(line))  # type: ignore
        return rows
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rows.append(_json.loads(s))
        return rows

def jslwrite(
    filepath: str,
    rows: Iterable[Any],
    backend: str = "json",
    append: bool = False,
) -> None:
    """
    Write NDJSON (one JSON object per line). Always compact; one trailing '\n' per row.
    If append=True, appends to the file.
    """
    if backend == "orjson":
        opt = 0
        opt |= _orjson.OPT_SERIALIZE_NUMPY  # type: ignore
        opt |= _orjson.OPT_APPEND_NEWLINE      # type: ignore

        mode = "ab" if append else "wb"
        with open(filepath, mode) as f:  # binary
            for obj in rows:
                f.write(_orjson.dumps(obj, option=opt))  # type: ignore
        return
    else:
        # std json path (compact; ensure_ascii=False for UTF-8)
        mode = "a" if append else "w"
        with open(filepath, mode, encoding="utf-8", newline="\n") as f:
            dumps = _json.dumps
            for obj in rows:
                def _np_default(o: Any):
                    if isinstance(o, (np.integer,)): return int(o)
                    if isinstance(o, (np.floating,)): return float(o)
                    if isinstance(o, (np.ndarray,)): return o.tolist()
                    raise TypeError

                s = dumps(
                    obj,
                    ensure_ascii=False,
                    separators=(",", ":"), # compact
                    default=_np_default,
                )
                f.write(s + "\n")