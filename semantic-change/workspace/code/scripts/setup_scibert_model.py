#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from zipfile import ZipFile
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def log(message: str) -> None:
    print(message, flush=True)


def detect_workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir_for(workspace_root: Path) -> Path:
    return workspace_root / "local" / "models" / "scibert_scivocab_uncased"


def bundled_zip_for(workspace_root: Path) -> Path:
    bundle_root = workspace_root.parent.parent
    return bundle_root / "shared-assets" / "models" / "scibert_uncased" / "scibert_bundle.zip"


def local_model_ready(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    try:
        AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        AutoModel.from_pretrained(model_dir, attn_implementation="eager")
    except Exception:
        return False
    return True


def materialize_local_model(model_dir: Path, bundled_zip: Path) -> None:
    if not bundled_zip.exists():
        raise FileNotFoundError(
            f"Missing bundled SciBERT archive at {bundled_zip}. "
            "The repo/source tree should provide shared-assets/models/scibert_uncased/scibert_bundle.zip."
        )

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    log(f"Unpacking bundled SciBERT archive from {bundled_zip}")
    with ZipFile(bundled_zip) as zf:
        zf.extractall(model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the local SciBERT directory expected by the semantic-change notebooks from the bundled base-model archive."
    )
    parser.add_argument(
        "--force-rematerialize",
        action="store_true",
        help="Rewrite the local SciBERT directory even if it already loads correctly.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = detect_workspace_root()
    model_dir = model_dir_for(workspace_root)
    bundled_zip = bundled_zip_for(workspace_root)

    log(f"Workspace root: {workspace_root}")
    log(f"Target local model directory: {model_dir}")
    log(f"Bundled SciBERT archive: {bundled_zip}")

    if local_model_ready(model_dir) and not args.force_rematerialize:
        log("Local SciBERT directory already loads correctly. Nothing to do.")
        return 0

    materialize_local_model(model_dir, bundled_zip=bundled_zip)

    log("Verifying the materialized local SciBERT directory ...")
    AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    AutoModel.from_pretrained(model_dir, attn_implementation="eager")
    log("SciBERT is ready for local notebook use.")
    log("You can now run `1.0.0-embeddings.ipynb` without a notebook-time model download.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
