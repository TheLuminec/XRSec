"""
download_datasets.py
=====================

This script reads a JSON catalog of dataset metadata (stored in
``datasets.json``) and attempts to download each dataset's files into
separate directories.  For every dataset entry, the script will:

1. Create a subdirectory under the chosen base directory using the
   dataset's name (with spaces and path separators replaced by
   underscores).
2. Write a human‑readable ``info.txt`` file into the dataset's
   directory that lists the dataset name, citation, sources, notes and
   any download instructions from the JSON.
3. Iterate over all download links provided in the JSON and fetch
   each file.  The script uses ``requests`` to stream the file
   contents to disk.  If a download fails (for example due to
   authentication requirements or network errors), the exception is
   recorded in the ``info.txt`` file so you can follow up manually.

Usage:
    python download_datasets.py [--json PATH] [--output-dir PATH]

Arguments:
    --json PATH       Path to the JSON file containing dataset
                      definitions.  Defaults to ``datasets.json`` in
                      the current working directory.
    --output-dir PATH Base directory where datasets will be saved.
                      Defaults to ``downloaded_datasets`` in the
                      current working directory.

Requirements:
    This script requires the ``requests`` package.  Install it via
    ``pip install requests`` if not already available.

Note:
    Some datasets require manual intervention to download.  For
    example, Google Drive and Aliyun links often prompt for sign‑in
    or terms of use.  In these cases, the script will catch the
    exception and note that manual download is required.  After
    downloading manually, place the files into the corresponding
    dataset folder.
"""

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

try:
    import requests  # type: ignore[import]
except ImportError as exc:
    sys.stderr.write(
        "The 'requests' library is required to run this script.\n"
        "Install it using 'pip install requests' and try again.\n"
    )
    raise


def sanitize_name(name: str) -> str:
    """Sanitize dataset name for use as a directory name."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def download_file(url: str, dest_path: Path) -> None:
    """Download a single file from ``url`` to ``dest_path``.

    If the destination file already exists, the download is skipped.
    Any exceptions raised by ``requests`` are propagated to the caller.
    """
    # If file already exists, skip download
    if dest_path.exists():
        print(f"[SKIP] {dest_path} already exists.")
        return

    print(f"[INFO] Downloading {url} ...")
    # Stream download the file
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    print(f"[OK] Saved to {dest_path}")


def write_info_file(dataset: dict, dataset_dir: Path, errors: list) -> None:
    """Write an info.txt file summarising the dataset metadata.

    Parameters
    ----------
    dataset : dict
        The dataset entry from the JSON.
    dataset_dir : Path
        Directory where the info file should be written.
    errors : list
        A list of strings describing any download errors to append to
        the info file.
    """
    info_path = dataset_dir / "info.txt"
    with open(info_path, "w", encoding="utf-8") as info:
        info.write(f"Name: {dataset.get('name', 'Unknown')}\n")
        info.write(f"Citation: {dataset.get('citation', 'N/A')}\n\n")
        info.write("Sources:\n")
        for src in dataset.get("sources", []):
            info.write(f" - {src}\n")
        info.write("\nNotes:\n")
        info.write(dataset.get("notes", "No notes provided.") + "\n\n")
        instructions = dataset.get("download_instructions")
        if instructions:
            info.write("Download Instructions:\n")
            info.write(instructions + "\n\n")
        if errors:
            info.write("Download Errors / Manual Actions Required:\n")
            for err in errors:
                info.write(f" - {err}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets defined in a JSON file.")
    parser.add_argument(
        "--json",
        type=str,
        default="datasets.json",
        help="Path to datasets.json (default: datasets.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="downloaded_datasets",
        help="Directory where datasets will be downloaded (default: downloaded_datasets)",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Whether or not to skip full downloads."
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    output_dir = Path(args.output_dir)
    skip_full = args.skip_full
    if not json_path.exists():
        print(f"Error: JSON file {json_path} does not exist.")
        sys.exit(1)
    # Create base output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset definitions
    with open(json_path, "r", encoding="utf-8") as fh:
        datasets = json.load(fh)

    for dataset in datasets:
        name = dataset.get("name", "Unnamed_Dataset")
        folder_name = sanitize_name(name)
        dataset_dir = output_dir / folder_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        if not skip_full:
            # Attempt to download each link
            for url in dataset.get("download_links", []):
                # Parse file name; if path is empty, create a default name
                parsed = urllib.parse.urlparse(url)
                filename = os.path.basename(parsed.path)
                # If there is no file name (e.g., Google Drive folder), skip download
                if not filename or filename == "":
                    errors.append(
                        f"Could not download from {url} automatically. The link may refer to a folder or require sign‑in. Please download manually."
                    )
                    continue
                dest_path = dataset_dir / filename
                try:
                    download_file(url, dest_path)
                except Exception as exc:
                    # Record error and continue with next file
                    err_msg = f"Failed to download {url}: {exc}"
                    print(f"[ERROR] {err_msg}")
                    errors.append(err_msg)

        # Write info.txt summarising dataset and any errors
        write_info_file(dataset, dataset_dir, errors)

    print("\nAll tasks completed. Check the output directory for downloaded datasets and info files.")


if __name__ == "__main__":
    main()