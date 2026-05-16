from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tam_fno.tam_fno_train import train_tam_fno


if __name__ == "__main__":
    train_tam_fno(SCRIPT_DIR)
