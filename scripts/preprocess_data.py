from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tam_fno_config import get_project_paths
from tam_fno_data import preprocess_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw TL/SSP data for TAM-FNO")
    parser.add_argument("--raw-tl", type=str, default="")
    parser.add_argument("--raw-ssp", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--split-manifest", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = get_project_paths(PROJECT_ROOT)
    raw_tl_path = Path(args.raw_tl) if args.raw_tl else paths.raw_data_dir / "TL.mat"
    raw_ssp_path = Path(args.raw_ssp) if args.raw_ssp else paths.raw_data_dir / "SSP.mat"
    output_dir = Path(args.output_dir) if args.output_dir else paths.data_dir
    split_manifest = (
        Path(args.split_manifest) if args.split_manifest else paths.split_manifest_path
    )

    result = preprocess_raw_data(
        raw_tl_path=raw_tl_path,
        raw_ssp_path=raw_ssp_path,
        output_dir=output_dir,
        split_manifest_path=split_manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
