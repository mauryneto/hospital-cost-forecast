"""Project paths.

``ROOT`` is where the code and versioned files live (configs, templates, checksum manifests). ``HOME`` is where data and
reports are written: the repository itself by default, or ``$COSTMODEL_HOME`` (used by Docker and the end-to-end
tests). Data under ``HOME/data`` is never committed.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("COSTMODEL_HOME", ROOT)).resolve()
DATA_DIR = HOME / "data"
REPORTS_DIR = HOME / "reports"
CONFIGS_DIR = ROOT / "configs"
CHECKSUMS_DIR = ROOT / "data" / "checksums"
