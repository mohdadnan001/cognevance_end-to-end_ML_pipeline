"""Data acquisition and loading utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from src import config

CSV_FILENAME = "diabetic_data.csv"
ZIP_FILENAME = "diabetes_130_us_hospitals.zip"
PROVENANCE_FILENAME = "provenance.json"


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(force: bool = False) -> Path:
    """Download and extract the UCI readmission dataset.

    Returns the path to ``diabetic_data.csv``. Existing files are reused
    unless ``force`` is True.
    """
    config.ensure_directories()
    csv_path = config.RAW_DIR / CSV_FILENAME
    provenance_path = config.RAW_DIR / PROVENANCE_FILENAME

    if csv_path.exists() and provenance_path.exists() and not force:
        print(f"[data] Using cached dataset: {csv_path}")
        return csv_path

    zip_path = config.RAW_DIR / ZIP_FILENAME
    if force or not zip_path.exists():
        print(f"[data] Downloading {config.DATASET_URL}")
        with requests.get(config.DATASET_URL, stream=True, timeout=180) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)

    print(f"[data] Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(config.RAW_DIR)

    candidates = list(config.RAW_DIR.rglob(CSV_FILENAME))
    if not candidates:
        raise FileNotFoundError(f"{CSV_FILENAME} was not found in {zip_path}")
    if candidates[0].resolve() != csv_path.resolve():
        shutil.copy2(candidates[0], csv_path)

    provenance = {
        "source_url": config.DATASET_URL,
        "dataset_page": config.DATASET_PAGE,
        "doi": config.DATASET_DOI,
        "license": config.DATASET_LICENSE,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256sum(csv_path),
        "rows": None,
        "columns": None,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[data] Saved dataset to {csv_path}")
    return csv_path


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw dataset, preserving the literal ``None`` category values."""
    if path is None:
        path = config.RAW_DIR / CSV_FILENAME
        if not Path(path).exists():
            path = download_dataset()

    frame = pd.read_csv(
        path,
        na_values=["?"],
        keep_default_na=False,
        low_memory=False,
    )
    return frame


def update_provenance(rows: int, columns: int) -> None:
    """Record the dataset shape in the provenance file."""
    provenance_path = config.RAW_DIR / PROVENANCE_FILENAME
    if not provenance_path.exists():
        return
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["rows"] = rows
    provenance["columns"] = columns
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
