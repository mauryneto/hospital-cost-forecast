"""A small vectorised reader for dBASE III ``.dbf`` files (the format inside DATASUS ``.dbc``).

A DBF is a header, a list of 32-byte field descriptors, then fixed-width records. Because every record has the same
width, the whole table is one 2-D byte array and each column is a slice of it: no per-row Python loop. This is ~50x
faster than a row-by-row reader on SIH files (about 1 s per month instead of ~30 s).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    offset: int  # inside the record, after the 1-byte deletion flag
    length: int


def parse_header(buf: bytes) -> tuple[int, int, int, list[Field]]:
    n_records, header_len, record_len = struct.unpack("<IHH", buf[4:12])
    fields, pos, offset = [], 32, 1
    while pos < header_len - 1 and buf[pos] != 0x0D:
        desc = buf[pos : pos + 32]
        name = desc[:11].split(b"\0", 1)[0].decode("ascii").strip()
        fields.append(Field(name, chr(desc[11]), offset, desc[16]))
        offset += desc[16]
        pos += 32
    if offset != record_len:
        raise ValueError(f"field widths add up to {offset}, header says records are {record_len} bytes")
    return n_records, header_len, record_len, fields


def read_dbf(buf: bytes, columns: list[str] | None = None, encoding: str = "latin-1") -> pd.DataFrame:
    """Read the given columns as text (stripped; empty -> <NA>). Deleted records are skipped."""
    n_records, header_len, record_len, fields = parse_header(buf)
    by_name = {f.name: f for f in fields}
    columns = columns or list(by_name)
    missing = [c for c in columns if c not in by_name]
    if missing:
        raise KeyError(f"columns not in file: {missing}")
    available = (len(buf) - header_len) // record_len
    n = min(n_records, available)
    records = np.frombuffer(buf, dtype=np.uint8, count=n * record_len, offset=header_len).reshape(n, record_len)
    keep = records[:, 0] != ord("*")
    records = records[keep]
    out = {}
    for name in columns:
        f = by_name[name]
        raw = np.ascontiguousarray(records[:, f.offset : f.offset + f.length]).view(f"S{f.length}").ravel()
        text = pd.Series(raw).str.decode(encoding, errors="replace").str.strip("\0 ").astype("string")
        out[name] = text.mask(text == "")
    return pd.DataFrame(out)
