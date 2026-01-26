#!/usr/bin/env python3
"""
Analyze cascade simulation dump files using cedar to extract vacancy/SIA information.

This script processes 1000 cascade simulations (N=1-100, M=1-10) and generates
defect information using the cedar post-processing utility.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# Configuration
CEDAR_EXECUTABLE = "/flare/Cascaide/romano/software/cedar_2024jan/cedar"
REFERENCE_FILE = "/flare/Cascaide/knight/lammps/W_Chen19_Jan2026/sc80/therm/dump.ref"
BASE_DUMP_DIR = "/flare/Cascaide/knight/lammps/W_Chen19_Jan2026/sc80/50keV/pka1"
OUTPUT_BASE_DIR = "/flare/Cascaide/romano/defect_analysis/sc80_50keV_pka1"

# Range of simulations
N_RANGE = range(1, 101)  # 1 to 100
M_RANGE = range(1, 11)   # 1 to 10


def check_prerequisites():
    """Check if required files and executables exist."""
    cedar_path = Path(CEDAR_EXECUTABLE)
    ref_path = Path(REFERENCE_FILE)

    if not cedar_path.exists():
        print(f"ERROR: Cedar executable not found at {CEDAR_EXECUTABLE}", file=sys.stderr)
        return False

    if not cedar_path.is_file() or not os.access(cedar_path, os.X_OK):
        print(f"ERROR: Cedar executable is not executable: {CEDAR_EXECUTABLE}", file=sys.stderr)
        return False

    if not ref_path.exists():
        print(f"ERROR: Reference file not found at {REFERENCE_FILE}", file=sys.stderr)
        return False

    return True


def cedar_defect_analysis(reference_dump, displaced_dump, output_dir=None, show_output=False):
    """
    Run cedar defect analysis on a displaced dump file.

    Parameters:
    -----------
    reference_dump : str or Path
        Path to the reference lattice dump file
    displaced_dump : str or Path
        Path to the displaced dump file to analyze
    output_dir : str or Path, optional
        Directory where cedar output files will be written (default: current directory)
    show_output : bool
        If True, stream cedar stdout/stderr to the console

    Returns:
    --------
    bool : True if successful, False otherwise
    """
    reference_dump = Path(reference_dump)
    displaced_dump = Path(displaced_dump)

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    # Check if input files exist
    if not reference_dump.exists():
        print(f"ERROR: Reference dump not found: {reference_dump}", file=sys.stderr)
        return False

    if not displaced_dump.exists():
        print(f"WARNING: Displaced dump not found: {displaced_dump}", file=sys.stderr)
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cedar command
    cmd = [
        CEDAR_EXECUTABLE,
        "-defect",
        str(reference_dump),
        "--fin",
        str(displaced_dump)
    ]

    # Run cedar in the output directory
    try:
        result = subprocess.run(
            cmd,
            cwd=str(output_dir),
            capture_output=not show_output,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"ERROR: Cedar failed for {displaced_dump}", file=sys.stderr)
            if not show_output and result.stderr:
                print(f"  STDERR: {result.stderr}", file=sys.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"ERROR: Cedar timed out for {displaced_dump}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Exception processing {displaced_dump}: {e}", file=sys.stderr)
        return False


def main():
    """Main processing loop."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process cascade simulations with cedar defect analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--n-start",
        type=int,
        default=1,
        help="Starting N value (default: 1)"
    )
    parser.add_argument(
        "--n-end",
        type=int,
        default=100,
        help="Ending N value (default: 100)"
    )
    parser.add_argument(
        "--m-start",
        type=int,
        default=1,
        help="Starting M value (default: 1)"
    )
    parser.add_argument(
        "--m-end",
        type=int,
        default=10,
        help="Ending M value (default: 10)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)"
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Stream cedar stdout/stderr while running"
    )

    args = parser.parse_args()

    # Check prerequisites
    if not args.dry_run:
        if not check_prerequisites():
            return 1

    # Statistics
    total = (args.n_end - args.n_start + 1) * (args.m_end - args.m_start + 1)
    processed = 0
    success = 0
    failed = 0
    skipped = 0

    start_time = datetime.now()

    print(f"Starting cascade analysis at {start_time}")
    print(f"Processing {total} simulations (N={args.n_start}-{args.n_end}, M={args.m_start}-{args.m_end})")
    print(f"Reference file: {REFERENCE_FILE}")
    print(f"Cedar executable: {CEDAR_EXECUTABLE}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("-" * 80)

    # Submit all tasks
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for n in range(args.n_start, args.n_end + 1):
            for m in range(args.m_start, args.m_end + 1):
                # Build paths for this cascade
                displaced_dump = Path(BASE_DUMP_DIR) / str(n) / str(m) / "dump.end.min"
                output_dir = Path(OUTPUT_BASE_DIR) / str(n) / str(m)

                if args.dry_run:
                    print(f"[DRY RUN] Would run cedar for N={n}, M={m}")
                    print(f"  Reference: {REFERENCE_FILE}")
                    print(f"  Displaced: {displaced_dump}")
                    print(f"  Output: {output_dir}")
                    processed += 1
                    success += 1
                else:
                    future = executor.submit(
                        cedar_defect_analysis,
                        REFERENCE_FILE,
                        displaced_dump,
                        output_dir,
                        args.show_output
                    )
                    futures.append(future)

        # Wait for completion and update progress from main thread
        for future in as_completed(futures):
            try:
                result = future.result()

                # Update counters
                processed += 1
                if result is True:
                    success += 1
                elif result is False:
                    failed += 1
                else:
                    skipped += 1

                # Print progress update from main thread
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"Progress: {processed}/{total} ({100*processed/total:.1f}%) "
                      f"- Success: {success}, Failed: {failed}, Skipped: {skipped} "
                      f"- Rate: {rate:.1f} sim/min, ETA: {eta:.1f} min")

            except Exception as e:
                processed += 1
                failed += 1
                print(f"ERROR: Unexpected exception in thread: {e}", file=sys.stderr)

    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("-" * 80)
    print(f"Analysis complete at {end_time}")
    print(f"Total time: {duration/60:.1f} minutes")
    print(f"Processed: {processed}/{total}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
