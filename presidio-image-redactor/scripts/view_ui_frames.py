#!/usr/bin/env python3
"""Find and view DICOM UI/settings frames one by one.

Scans a folder for Secondary Capture files with InstanceNumber in (0, 1) and
ConversionType == WSD, then displays each one for visual review.

Usage:
    python view_ui_frames.py /path/to/dicom/root
    python view_ui_frames.py /path/to/dicom/root --save-dir /tmp/ui_frames
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

SC_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"


def find_ui_frames(root_dir: Path) -> list:
    """Find all DICOM files matching UI frame criteria."""
    matches = []
    for dcm_file in sorted(root_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
        except Exception:
            continue

        sop_uid = str(getattr(ds, "SOPClassUID", ""))
        if sop_uid != SC_SOP_CLASS_UID:
            continue

        instance_num = getattr(ds, "InstanceNumber", None)
        conversion_type = getattr(ds, "ConversionType", "")
        image_type = list(getattr(ds, "ImageType", []))
        manufacturer = getattr(ds, "Manufacturer", "").upper()

        is_ui = False
        if "GE" in manufacturer:
            is_ui = conversion_type == "WSD" and instance_num in (0, 1, 256)
        elif "SIEMENS" in manufacturer:
            is_ui = "EXAMPROTOCOL" in image_type
        elif "PHILIPS" in manufacturer:
            is_ui = conversion_type == "WSD" and image_type == ["DERIVED", "PRIMARY"]
        else:
            is_ui = conversion_type == "WSD" and instance_num in (0, 1)

        if is_ui:
            matches.append(dcm_file)

    return matches


def get_accession_dirs(root_dir: Path) -> list:
    """Get all accession-level subdirectories."""
    return sorted([d for d in root_dir.iterdir() if d.is_dir()])


def load_image(dcm_path: Path) -> Image.Image:
    """Load a DICOM file as a PIL Image."""
    ds = pydicom.dcmread(str(dcm_path))
    pixel_arr = ds.pixel_array

    is_greyscale = getattr(ds, "PhotometricInterpretation", "") in (
        "MONOCHROME1", "MONOCHROME2",
    )

    if is_greyscale:
        while pixel_arr.ndim > 2:
            pixel_arr = pixel_arr[0]
        if pixel_arr.dtype != np.uint8:
            arr_min, arr_max = pixel_arr.min(), pixel_arr.max()
            if arr_max > arr_min:
                pixel_arr = ((pixel_arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
            else:
                pixel_arr = np.zeros_like(pixel_arr, dtype=np.uint8)
        return Image.fromarray(pixel_arr, mode="L")
    else:
        while pixel_arr.ndim > 3:
            pixel_arr = pixel_arr[0]
        if pixel_arr.dtype != np.uint8:
            arr_min, arr_max = pixel_arr.min(), pixel_arr.max()
            if arr_max > arr_min:
                pixel_arr = ((pixel_arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
            else:
                pixel_arr = np.zeros_like(pixel_arr, dtype=np.uint8)
        return Image.fromarray(pixel_arr, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Find and view DICOM UI frames.")
    parser.add_argument("input", type=Path, help="Root directory to scan")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Save frames as PNGs to this directory instead of displaying")
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.input} for UI frames...")
    print()

    # Find all UI frames
    frames = find_ui_frames(args.input)

    # Map frames to their accession (parent dir)
    accessions = get_accession_dirs(args.input)
    frames_by_accession = {}
    for f in frames:
        # Walk up to find which accession dir this belongs to
        for acc in accessions:
            try:
                f.relative_to(acc)
                frames_by_accession.setdefault(acc.name, []).append(f)
                break
            except ValueError:
                continue

    accessions_with = [a.name for a in accessions if a.name in frames_by_accession]
    accessions_without = [a.name for a in accessions if a.name not in frames_by_accession]

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total accessions:          {len(accessions)}")
    print(f"With UI frame detected:    {len(accessions_with)}")
    print(f"Without UI frame:          {len(accessions_without)}")
    print(f"Total UI frames found:     {len(frames)}")
    print()

    if accessions_without:
        print("Accessions missing UI frames:")
        for name in accessions_without:
            print(f"  - {name}")
        print()

    print("UI frames per accession:")
    for acc in accessions:
        count = len(frames_by_accession.get(acc.name, []))
        marker = "  OK" if count > 0 else "  ** MISSING **"
        print(f"  {acc.name}: {count} frame(s){marker}")
    print("=" * 60)
    print()

    if not frames:
        return

    # Save or display
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for i, dcm_path in enumerate(frames):
            study = dcm_path.parent.name
            try:
                img = load_image(dcm_path)
                out_path = args.save_dir / f"{study}_{dcm_path.stem}.png"
                img.save(str(out_path))
                print(f"  [{i+1}/{len(frames)}] Saved: {out_path.name}")
            except Exception as e:
                print(f"  [{i+1}/{len(frames)}] Could not save {dcm_path.name}: {e}")
        print(f"\nAll frames saved to {args.save_dir}")
    else:
        for i, dcm_path in enumerate(frames):
            study = dcm_path.parent.name
            print(f"[{i+1}/{len(frames)}] {study}/{dcm_path.name}")
            try:
                img = load_image(dcm_path)
                img.show()
            except Exception as e:
                print(f"  Could not display: {e}")
                continue

            resp = input("  Press Enter for next, 'q' to quit: ").strip().lower()
            if resp == "q":
                break


if __name__ == "__main__":
    main()
