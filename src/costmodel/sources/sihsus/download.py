"""Download SIH/SUS "AIH reduzida" (RD) files from the DATASUS FTP and verify them against a checksum manifest.

Each file is ``RD{UF}{YYMM}.dbc``: one state, one *processing* month (competencia). The raw files are never committed;
only ``data/checksums/sihsus.json`` is. The first download of a file records its SHA-256; every later download (or a
re-run on an existing file) must match it, so anyone reproducing the project knows they are training on the same bytes.
"""

from __future__ import annotations

import ftplib
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from costmodel.paths import CHECKSUMS_DIR, DATA_DIR

FTP_HOST = "ftp.datasus.gov.br"
FTP_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados"

SIH_DIR = DATA_DIR / "sihsus"
RAW_DIR = SIH_DIR / "raw"
MANIFEST = CHECKSUMS_DIR / "sihsus.json"  # versioned with the code


def rd_filename(uf: str, year: int, month: int) -> str:
    return f"RD{uf.upper()}{year % 100:02d}{month:02d}.dbc"


def months(first: str, last: str) -> Iterable[tuple[int, int]]:
    """Inclusive range of (year, month) between two "YYYY-MM" strings."""
    year, month = map(int, first.split("-"))
    end = tuple(map(int, last.split("-")))
    while (year, month) <= end:
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(manifest: dict[str, str], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n", encoding="utf-8", newline="\n")


class ChecksumMismatch(RuntimeError):
    pass


def verify_or_record(name: str, path: Path, manifest: dict[str, str]) -> str:
    """Return the file's hash; record it if new, raise if it differs from the manifest."""
    digest = sha256(path)
    expected = manifest.get(name)
    if expected is None:
        manifest[name] = digest
    elif expected != digest:
        raise ChecksumMismatch(f"{name}: expected {expected}, got {digest}. DATASUS may have republished the file.")
    return digest


def download(uf: str, first: str, last: str, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Download processing months ``first``..``last`` ("YYYY-MM") for one state."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    known = dict(manifest)
    paths: list[Path] = []
    with ftplib.FTP(FTP_HOST, timeout=120) as ftp:
        ftp.login()
        ftp.cwd(FTP_DIR)
        for year, month in months(first, last):
            name = rd_filename(uf, year, month)
            path = raw_dir / name
            if not path.exists():
                tmp = path.with_suffix(".part")
                with tmp.open("wb") as f:
                    ftp.retrbinary(f"RETR {name}", f.write)
                tmp.replace(path)
                print(f"downloaded {name} ({path.stat().st_size / 1e6:.1f} MB)")
            verify_or_record(name, path, manifest)
            paths.append(path)
    if manifest != known:  # only new files change the versioned manifest; a reproduction run never writes it
        save_manifest(manifest)
        print(f"recorded {len(manifest) - len(known)} new checksum(s) in {MANIFEST.name}")
    print(f"{len(paths)} files match the checksum manifest")
    return paths
