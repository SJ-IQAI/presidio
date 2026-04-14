#!/usr/bin/env python3
"""De-identify DICOM images by redacting burned-in text and optionally scrubbing PHI metadata tags.

Examples:
    # Redact a single DICOM directory
    python redact_dicom.py /path/to/dicoms /path/to/output

    # Redact all subdirectories under a root (batch mode)
    python redact_dicom.py /path/to/root /path/to/output --batch

    # Use EasyOCR instead of the default Tesseract OCR
    python redact_dicom.py /path/to/dicoms /path/to/output --ocr easyocr

    # Also scrub PHI metadata tags from the output files
    python redact_dicom.py /path/to/dicoms /path/to/output --scrub-metadata

    # Disable automatic UI frame blackout
    python redact_dicom.py /path/to/dicoms /path/to/output --no-blackout-ui-frames
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom

from presidio_image_redactor import DicomImageRedactorEngine

# DICOM tags commonly containing PHI
PHI_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientAge",
    "PatientSex", "AccessionNumber", "InstitutionName",
    "ReferringPhysicianName", "PerformingPhysicianName",
    "OperatorsName", "StudyDate", "SeriesDate", "AcquisitionDate",
    "ContentDate", "StudyTime", "SeriesTime", "AcquisitionTime",
    "ContentTime", "StationName", "StudyDescription",
    "SeriesDescription", "InstitutionalDepartmentName",
    "DeviceSerialNumber", "ProtocolName", "StudyID",
]


# Secondary Capture Image Storage SOP Class UID
SC_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"


def is_ui_frame(ds: pydicom.Dataset) -> bool:
    """Detect whether a DICOM file is a UI/settings frame using metadata.

    Vendor-specific rules:
      GE:      SC + ConversionType WSD + InstanceNumber in (0, 1, 256)
      Siemens: SC + ImageType contains EXAMPROTOCOL
      Philips: SC + ConversionType WSD + ImageType == ['DERIVED', 'PRIMARY']

    :param ds: A pydicom Dataset (header only is sufficient).
    :return: True if the file appears to be a UI frame.
    """
    sop_uid = str(getattr(ds, "SOPClassUID", ""))
    if sop_uid != SC_SOP_CLASS_UID:
        return False

    instance_num = getattr(ds, "InstanceNumber", None)
    conversion_type = getattr(ds, "ConversionType", "")
    image_type = list(getattr(ds, "ImageType", []))
    manufacturer = getattr(ds, "Manufacturer", "").upper()

    # GE: patient-info screen — SC + WSD + low instance number
    if "GE" in manufacturer:
        return conversion_type == "WSD" and instance_num in (0, 1, 256)

    # Siemens: exam protocol sheet — SC + EXAMPROTOCOL in ImageType
    if "SIEMENS" in manufacturer:
        return "EXAMPROTOCOL" in image_type

    # Philips: dose report — SC + WSD + DERIVED/PRIMARY only
    if "PHILIPS" in manufacturer:
        return conversion_type == "WSD" and image_type == ["DERIVED", "PRIMARY"]

    # Unknown vendor: fall back to generic SC + low instance + WSD
    return conversion_type == "WSD" and instance_num in (0, 1)


def blackout_ui_frames(output_dir: Path) -> int:
    """Scan output DICOM files and black out any that are UI/settings frames.

    Detection is metadata-based (no OCR needed): Secondary Capture + InstanceNumber 0
    + ConversionType WSD.

    :param output_dir: Directory containing redacted DICOM files.
    :return: Number of frames blacked out.
    """
    print(f"  Scanning for UI frames (metadata-based)...")
    count = 0
    for dcm_file in sorted(output_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dcm_file))
        except Exception as e:
            print(f"  Skipping {dcm_file.name}: {e}")
            continue

        if is_ui_frame(ds):
            try:
                pixel_arr = ds.pixel_array
                ds.PixelData = np.zeros_like(pixel_arr).tobytes()
                # Switch to uncompressed transfer syntax so pydicom
                # doesn't complain about raw bytes in a compressed TS
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                ds.is_implicit_VR = False
                ds.is_little_endian = True
                ds.save_as(str(dcm_file))
                print(f"  Blacked out UI frame: {dcm_file.name}")
                count += 1
            except Exception as e:
                print(f"  Could not black out {dcm_file.name}: {e}")

    return count


def build_engine(ocr_type: str) -> DicomImageRedactorEngine:
    """Build a DicomImageRedactorEngine with the chosen OCR backend."""
    if ocr_type == "easyocr":
        from presidio_image_redactor.easyocr_engine import EasyOCREngine
        from presidio_image_redactor.image_analyzer_engine import ImageAnalyzerEngine

        ocr = EasyOCREngine()
        image_analyzer = ImageAnalyzerEngine(ocr=ocr)
        return DicomImageRedactorEngine(image_analyzer_engine=image_analyzer)

    return DicomImageRedactorEngine()


def scrub_metadata(output_dir: Path) -> None:
    """Blank PHI metadata tags in all DICOM files under output_dir."""
    for dcm_file in output_dir.rglob("*.dcm"):
        ds = pydicom.dcmread(str(dcm_file))
        for tag in PHI_TAGS:
            if hasattr(ds, tag):
                setattr(ds, tag, "")
        ds.save_as(str(dcm_file))


def redact_directory(
    engine: DicomImageRedactorEngine,
    input_path: Path,
    output_path: Path,
    ocr_threshold: float,
    fill: str,
    use_metadata: bool,
) -> None:
    """Redact a single DICOM directory."""
    output_path.mkdir(parents=True, exist_ok=True)
    engine.redact_from_directory(
        str(input_path),
        str(output_path),
        fill=fill,
        ocr_kwargs={"ocr_threshold": ocr_threshold},
        use_metadata=use_metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="De-identify DICOM images by redacting burned-in text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Input DICOM directory (or root directory in batch mode)")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Process each subdirectory of INPUT as a separate DICOM set")
    parser.add_argument("--ocr", choices=["tesseract", "easyocr"], default="tesseract",
                        help="OCR backend to use (default: tesseract)")
    parser.add_argument("--ocr-threshold", type=float, default=0.4,
                        help="OCR confidence threshold (default: 0.4)")
    parser.add_argument("--fill", choices=["contrast", "background"], default="background",
                        help="Redaction box fill mode (default: background)")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Disable metadata-based redaction of burned-in text")
    parser.add_argument("--scrub-metadata", action="store_true",
                        help="Also blank PHI metadata tags in the output DICOM files")
    parser.add_argument("--no-blackout-ui-frames", action="store_true",
                        help="Disable automatic detection and blackout of UI/settings frames (e.g. GE patient info screens)")
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    engine = build_engine(args.ocr)
    failed = []

    if args.batch:
        subdirs = sorted(d for d in args.input.iterdir() if d.is_dir())
        if not subdirs:
            print(f"No subdirectories found in {args.input}", file=sys.stderr)
            sys.exit(1)

        for subdir in subdirs:
            output_dir = args.output / subdir.name
            print(f"Processing: {subdir.name}")
            try:
                redact_directory(engine, subdir, output_dir, args.ocr_threshold, args.fill, not args.no_metadata)
                if not args.no_blackout_ui_frames:
                    n = blackout_ui_frames(output_dir)
                    if n:
                        print(f"  Blacked out {n} UI frame(s)")
                if args.scrub_metadata:
                    scrub_metadata(output_dir)
                print(f"  Done: {subdir.name}")
            except Exception as e:
                print(f"  FAILED: {subdir.name} — {e}")
                failed.append(subdir.name)
    else:
        print(f"Processing: {args.input}")
        try:
            redact_directory(engine, args.input, args.output, args.ocr_threshold, args.fill, not args.no_metadata)
            if not args.no_blackout_ui_frames:
                n = blackout_ui_frames(args.output)
                if n:
                    print(f"Blacked out {n} UI frame(s)")
            if args.scrub_metadata:
                scrub_metadata(args.output)
            print("Done")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    if failed:
        print(f"\n{len(failed)} directories failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
