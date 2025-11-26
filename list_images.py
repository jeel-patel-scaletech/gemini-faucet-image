#!/usr/bin/env python3
"""Script to list files in input_images directory and output JSON with filenames as keys."""

import json
from pathlib import Path

INPUT_DIR = "input_images"
METADATA_FILE = "image_metadata.json"
OUTPUT_FILE = "image_list.json"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_metadata(metadata_file: str) -> dict:
    """Load metadata from JSON file."""
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        print(f"Warning: Metadata file '{metadata_file}' not found. Using empty metadata.")
        return {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing metadata file: {e}. Using empty metadata.")
        return {}


def main():
    input_path = Path(INPUT_DIR)
    
    if not input_path.exists():
        print(f"Error: Directory '{INPUT_DIR}' does not exist.")
        return
    
    # Load existing metadata
    metadata_map = load_metadata(METADATA_FILE)
    
    # Get all image files with their metadata
    image_files = {}
    for file_path in sorted(input_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            # Use metadata if available, otherwise empty dict
            image_files[file_path.name] = metadata_map.get(file_path.name, {})
    
    # Write to JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(image_files, f, indent=2, ensure_ascii=False)
    
    found_with_metadata = sum(1 for v in image_files.values() if v)
    print(f"Found {len(image_files)} image files.")
    print(f"  - {found_with_metadata} files have metadata")
    print(f"  - {len(image_files) - found_with_metadata} files without metadata")
    print(f"JSON output written to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()

