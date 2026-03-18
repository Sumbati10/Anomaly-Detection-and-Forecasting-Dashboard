from __future__ import annotations

import argparse
from pathlib import Path

from src.services.pipeline import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args()

    train_and_save(artifacts_dir=Path(args.artifacts_dir), contamination=args.contamination)


if __name__ == "__main__":
    main()
