import struct

import pandas as pd
import pytest

from costmodel.sources.sihsus.dbf import read_dbf


def make_dbf(fields: list[tuple[str, str, int]], rows: list[tuple[str, ...]], deleted: set[int] = frozenset()) -> bytes:
    record_len = 1 + sum(length for _, _, length in fields)
    header_len = 32 + 32 * len(fields) + 1
    header = struct.pack("<B3xIHH20x", 3, len(rows), header_len, record_len)
    descs = b"".join(
        name.encode().ljust(11, b"\0") + kind.encode() + b"\0" * 4 + bytes([length, 0]) + b"\0" * 14
        for name, kind, length in fields
    )
    body = b"".join(
        (b"*" if i in deleted else b" ")
        + b"".join(v.encode("latin-1").ljust(length)[:length] for v, (_, _, length) in zip(row, fields, strict=True))
        for i, row in enumerate(rows)
    )
    return header + descs + b"\r" + body + b"\x1a"


FIELDS = [("N_AIH", "C", 13), ("DIAG", "C", 4), ("VAL_TOT", "N", 8)]


def test_reads_columns_strips_and_marks_empty_as_missing():
    buf = make_dbf(FIELDS, [("3317107848703", "K650", "  452.57"), ("3317106692284", "", "     0.0")])
    df = read_dbf(buf)
    assert df["N_AIH"].tolist() == ["3317107848703", "3317106692284"]
    assert df["DIAG"].isna().tolist() == [False, True]
    assert pd.to_numeric(df["VAL_TOT"]).tolist() == [452.57, 0.0]


def test_skips_deleted_records_and_selects_columns():
    buf = make_dbf(FIELDS, [("1", "A01", "1"), ("2", "B02", "2"), ("3", "Ç03", "3")], deleted={1})
    df = read_dbf(buf, ["DIAG"])
    assert list(df.columns) == ["DIAG"]
    assert df["DIAG"].tolist() == ["A01", "Ç03"]  # latin-1 decoded


def test_missing_column_raises():
    with pytest.raises(KeyError):
        read_dbf(make_dbf(FIELDS, [("1", "A01", "1")]), ["NOPE"])
