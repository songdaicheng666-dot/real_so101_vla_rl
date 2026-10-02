"""Scoped Hugging Face Datasets cache helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def temporary_hf_datasets_cache() -> Iterator[Path]:
    """Use and remove an isolated cache for one dataset validation run.

    Hugging Face Datasets materializes local Parquet image columns as Arrow
    files. A fresh fingerprint is produced as a growing recording changes, so
    using the process-wide cache for repeated validation would retain one large
    copy per checkpoint.
    """

    previous_environment = os.environ.get("HF_DATASETS_CACHE")
    datasets_module: Any | None = None
    previous_runtime_cache: Any | None = None
    with tempfile.TemporaryDirectory(prefix="so101-validator-hf-") as directory:
        cache_path = Path(directory)
        os.environ["HF_DATASETS_CACHE"] = str(cache_path)
        try:
            try:
                import datasets
            except ImportError:
                pass
            else:
                datasets_module = datasets
                previous_runtime_cache = datasets.config.HF_DATASETS_CACHE
                datasets.config.HF_DATASETS_CACHE = str(cache_path)
            yield cache_path
        finally:
            if datasets_module is not None:
                datasets_module.config.HF_DATASETS_CACHE = previous_runtime_cache
            if previous_environment is None:
                os.environ.pop("HF_DATASETS_CACHE", None)
            else:
                os.environ["HF_DATASETS_CACHE"] = previous_environment
