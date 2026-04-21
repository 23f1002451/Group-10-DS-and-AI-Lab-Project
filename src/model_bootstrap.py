"""
Model checkpoint bootstrap utilities.

Checks whether models/final_model.pt exists at startup and optionally
downloads it from a remote URL if it is missing.

This module is the recommended solution for Streamlit Community Cloud
deployments where storing a 300 MB+ PyTorch checkpoint directly in the
Git repository is not practical.

Usage:
    from src.model_bootstrap import check_or_download

    ok, message = check_or_download(
        checkpoint_path="models/final_model.pt",
        download_url="https://example.com/final_model.pt",  # optional
    )
    if not ok:
        print(message)  # actionable error text
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple


def check_or_download(
    checkpoint_path: str,
    download_url: str = "",
) -> Tuple[bool, str]:
    """
    Ensure `checkpoint_path` exists on disk.

    Resolution order:
      1. File already exists  → return (True, informational message)
      2. `download_url` given → attempt download, return result
      3. Neither              → return (False, human-readable instructions)

    Args:
        checkpoint_path: Absolute or relative path to the .pt file.
        download_url:    Optional HTTPS URL to fetch the file from.
                         Supports direct links (e.g., Google Drive export URLs,
                         HuggingFace Hub raw file URLs, GitHub Releases assets).

    Returns:
        Tuple of (success: bool, message: str)
    """
    path = Path(checkpoint_path)

    # ── Case 1: checkpoint already present ───────────────────────────────
    if path.exists():
        size_mb = path.stat().st_size / (1024 ** 2)
        return True, f"Checkpoint found at '{path}' ({size_mb:.1f} MB)"

    # ── Case 2: try to download ───────────────────────────────────────────
    if download_url.strip():
        return _download_checkpoint(path, download_url.strip())

    # ── Case 3: not found, no download URL ───────────────────────────────
    instructions = (
        f"Model checkpoint not found at '{path}'.\n\n"
        "To fix this, choose one of the following options:\n\n"
        "Option A — Train locally:\n"
        "  python src/train.py \\\n"
        "    --train-data data/small/train.json \\\n"
        "    --val-data   data/small/validation.json \\\n"
        "    --output-dir models \\\n"
        "    --epochs 5\n\n"
        "Option B — Auto-download (Streamlit Cloud):\n"
        "  Set MODEL_DOWNLOAD_URL in .streamlit/secrets.toml to a direct\n"
        "  download link for final_model.pt (e.g., a GitHub Release asset,\n"
        "  a HuggingFace Hub raw URL, or a Google Drive export URL).\n\n"
        "Option C — Manual placement:\n"
        "  Copy final_model.pt into the models/ directory."
    )
    return False, instructions


# ── Private helpers ───────────────────────────────────────────────────────

def _download_checkpoint(dest: Path, url: str) -> Tuple[bool, str]:
    """Download a checkpoint file from `url` to `dest`."""
    import urllib.request

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[model_bootstrap] Downloading checkpoint from:\n  {url}")
        print(f"[model_bootstrap] Saving to: {dest}")

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                downloaded = min(block_num * block_size, total_size)
                pct = downloaded / total_size * 100
                mb_done = downloaded / (1024 ** 2)
                mb_total = total_size / (1024 ** 2)
                sys.stdout.write(
                    f"\r  Progress: {pct:5.1f}%  ({mb_done:.1f}/{mb_total:.1f} MB)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar

        size_mb = dest.stat().st_size / (1024 ** 2)
        return True, (
            f"Checkpoint downloaded successfully to '{dest}' ({size_mb:.1f} MB)"
        )

    except Exception as exc:
        # Clean up partial file if download failed
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return False, (
            f"Download failed from '{url}'.\n"
            f"Error: {exc}\n\n"
            "Please download the checkpoint manually and place it at:\n"
            f"  {dest}"
        )
