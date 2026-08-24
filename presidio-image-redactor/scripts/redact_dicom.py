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
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom

from presidio_image_redactor import DicomImageRedactorEngine


def log(msg: str) -> None:
    """Print with timestamp, flushed immediately."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def progress_bar(current: int, total: int, elapsed: float, label: str = "") -> None:
    """Draw a single-line dynamic progress bar to stderr (terminal only).

    Uses carriage return to overwrite the previous line. Falls back to plain
    logging when stderr is not a TTY (e.g., piped to file or tee).
    """
    if total <= 0:
        return
    pct = current / total
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0
    bar_width = 30
    filled = int(bar_width * pct)
    bar = "#" * filled + "-" * (bar_width - filled)
    line = (
        f"  [{bar}] {current}/{total} ({pct*100:.1f}%)  "
        f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m  {label}"
    )
    # Pad to clear previous content, overwrite same line
    sys.stderr.write("\r\033[K" + line)
    sys.stderr.flush()


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
        # Skip macOS AppleDouble sidecar files
        if dcm_file.name.startswith("._"):
            continue
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


# ---------------------------------------------------------------------------
# Multiprocessing support: each worker loads its own engine once via
# _worker_init, then processes assigned accessions end-to-end.
# ---------------------------------------------------------------------------

_worker_engine = None  # set once per worker process


def _worker_init(ocr_type: str) -> None:
    """Initialize a per-worker engine. Called once when the worker process starts."""
    global _worker_engine
    _worker_engine = build_engine(ocr_type)


def _process_one_accession(
    acc_str: str,
    input_root_str: str,
    output_root_str: str,
    ocr_threshold: float,
    fill: str,
    use_metadata: bool,
    no_blackout_ui_frames: bool,
    scrub_meta: bool,
) -> tuple:
    """Worker task: redact a single accession end-to-end.

    :return: (rel_path_str, success: bool, msg: str, elapsed_s: float, ui_blackout_count: int)
    """
    acc = Path(acc_str)
    input_root = Path(input_root_str)
    output_root = Path(output_root_str)
    rel = acc.relative_to(input_root)
    output_parent = (output_root / rel).parent
    output_dir = output_root / rel
    t0 = time.time()
    ui_count = 0
    try:
        redact_directory(_worker_engine, acc, output_parent, ocr_threshold, fill, use_metadata)
        if not no_blackout_ui_frames:
            ui_count = blackout_ui_frames(output_dir)
        if scrub_meta:
            scrub_metadata(output_dir)
        return (str(rel), True, "", time.time() - t0, ui_count)
    except Exception as e:
        return (str(rel), False, str(e), time.time() - t0, ui_count)


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
    parser.add_argument("--skip-prefix", action="append", default=[],
                        help="Skip subdirectories (at any level) whose name starts with this prefix. "
                             "Can be passed multiple times (e.g. --skip-prefix 00SP)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4). "
                             "Use 1 for serial processing. Each worker loads its own "
                             "~500MB EasyOCR model (~2GB RAM total for 4 workers).")
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Warn if destination is non-empty; prompt to clean or abort
    if args.output.exists() and any(args.output.iterdir()):
        import subprocess
        log(f"WARNING: Destination {args.output} is not empty.")
        resp = input("  [D]elete existing contents, [c]ontinue anyway, or [a]bort? ").strip().lower()
        if resp == "d":
            log(f"Removing {args.output}...")
            # Use rm -rf for robustness against macOS ._ sidecar files on external drives
            result = subprocess.run(["rm", "-rf", str(args.output)], capture_output=True, text=True)
            if result.returncode != 0:
                log(f"  rm warning: {result.stderr.strip()}")
            args.output.mkdir(parents=True, exist_ok=True)
            log("Destination cleaned.")
        elif resp == "a":
            log("Aborted.")
            sys.exit(0)
        else:
            log("Continuing with existing contents (may cause 'destination exists' errors).")

    workers = max(1, args.workers)
    # Only build an engine in the main process for serial / non-batch mode.
    # For parallel batch mode, each worker builds its own engine in _worker_init.
    engine = None
    if workers == 1 or not args.batch:
        engine = build_engine(args.ocr)
    failed = []

    if args.batch:
        # Collect all leaf directories containing .dcm files (accessions),
        # preserving their relative path under args.input.
        top_subdirs = sorted(d for d in args.input.iterdir() if d.is_dir())
        if not top_subdirs:
            print(f"No subdirectories found in {args.input}", file=sys.stderr)
            sys.exit(1)

        # Build list of (accession_dir, relative_path_under_input)
        accessions = []
        for top in top_subdirs:
            # If top itself contains dcms directly, treat it as an accession
            if any(top.glob("*.dcm")) or any(top.glob("*.DCM")):
                accessions.append(top)
            else:
                for acc in sorted(top.iterdir()):
                    if acc.is_dir():
                        accessions.append(acc)

        # Apply --skip-prefix filter (matches against directory name)
        if args.skip_prefix:
            filtered = []
            skipped = []
            for acc in accessions:
                if any(acc.name.startswith(p) for p in args.skip_prefix):
                    skipped.append(acc)
                else:
                    filtered.append(acc)
            if skipped:
                log(f"Skipping {len(skipped)} accession(s) matching prefix {args.skip_prefix}:")
                for s in skipped:
                    log(f"  - {s.relative_to(args.input)}")
            accessions = filtered

        total = len(accessions)
        log(f"Found {total} accession(s) to process")
        log(f"Output: {args.output}")
        log(f"OCR: {args.ocr}  threshold={args.ocr_threshold}  fill={args.fill}")
        log(f"Blackout UI frames: {not args.no_blackout_ui_frames}")
        log(f"Workers: {workers}")
        print()

        batch_start = time.time()

        if workers <= 1:
            # --- Serial mode ---
            for i, acc in enumerate(accessions, 1):
                rel = acc.relative_to(args.input)
                output_parent = (args.output / rel).parent
                output_dir = args.output / rel
                n_files = sum(1 for _ in acc.rglob("*.dcm")) + sum(1 for _ in acc.rglob("*.DCM"))
                progress_bar(i - 1, total, time.time() - batch_start, f"starting {rel} ({n_files} files)")
                print()
                log(f"[{i}/{total}] Starting: {rel}  ({n_files} files)")
                t0 = time.time()
                try:
                    redact_directory(engine, acc, output_parent, args.ocr_threshold, args.fill, not args.no_metadata)
                    if not args.no_blackout_ui_frames:
                        n = blackout_ui_frames(output_dir)
                        if n:
                            log(f"  -> Blacked out {n} UI frame(s)")
                    if args.scrub_metadata:
                        scrub_metadata(output_dir)
                    elapsed = time.time() - t0
                    total_elapsed = time.time() - batch_start
                    remaining = (total - i) * (total_elapsed / i) if i > 0 else 0
                    log(f"[{i}/{total}] Done: {rel}  ({elapsed:.1f}s, ETA {remaining/60:.1f}m)")
                except Exception as e:
                    log(f"[{i}/{total}] FAILED: {rel} — {e}")
                    failed.append(str(rel))
                print()
        else:
            # --- Parallel mode: ProcessPoolExecutor with per-worker engine ---
            log(f"Spawning {workers} worker(s); each will load an EasyOCR model (~30s startup)...")
            ctx = multiprocessing.get_context("spawn")
            task_args = [
                (
                    str(acc),
                    str(args.input),
                    str(args.output),
                    args.ocr_threshold,
                    args.fill,
                    not args.no_metadata,
                    args.no_blackout_ui_frames,
                    args.scrub_metadata,
                )
                for acc in accessions
            ]
            completed = 0
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(args.ocr,),
            ) as executor:
                futures = {
                    executor.submit(_process_one_accession, *targs): targs[0]
                    for targs in task_args
                }
                for future in as_completed(futures):
                    completed += 1
                    try:
                        rel, success, msg, elapsed, ui_count = future.result()
                    except Exception as e:
                        rel = Path(futures[future]).name
                        success, msg, elapsed, ui_count = False, str(e), 0.0, 0
                    total_elapsed = time.time() - batch_start
                    remaining = (total - completed) * (total_elapsed / completed) if completed > 0 else 0
                    if success:
                        extra = f", blacked out {ui_count} UI frame(s)" if ui_count else ""
                        log(f"[{completed}/{total}] Done: {rel}  ({elapsed:.1f}s{extra}, ETA {remaining/60:.1f}m)")
                    else:
                        log(f"[{completed}/{total}] FAILED: {rel} — {msg}")
                        failed.append(rel)
                    progress_bar(completed, total, total_elapsed, f"done {rel}")

        total_elapsed = time.time() - batch_start
        progress_bar(total, total, total_elapsed, "complete")
        print()
        log(f"Batch complete: {total} accession(s) in {total_elapsed/60:.1f}m ({len(failed)} failed)")
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
