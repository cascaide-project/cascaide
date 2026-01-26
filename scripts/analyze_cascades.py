#!/usr/bin/env python3
"""
Analyze cascade simulation dump files using cedar to extract vacancy/SIA information.

This script processes 1000 cascade simulations (N=1-100, M=1-10) and generates
defect information using the cedar post-processing utility.
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


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


def process_cascade(n, m, dry_run=False, show_output=False):
    """
    Process a single cascade simulation.

    Parameters:
    -----------
    n : int
        Starting state number (1-100)
    m : int
        PKA angle number (1-10)
    dry_run : bool
        If True, only print commands without executing
    show_output : bool
        If True, stream cedar stdout/stderr to the console

    Returns:
    --------
    bool : True if successful, False otherwise
    """
    # Input file path
    dump_file = Path(BASE_DUMP_DIR) / str(n) / str(m) / "dump.end.min"

    # Output directory
    output_dir = Path(OUTPUT_BASE_DIR) / str(n) / str(m)

    # Check if input file exists
    if not dump_file.exists():
        print(f"WARNING: Dump file not found: {dump_file}", file=sys.stderr)
        return False

    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cedar command
    cmd = [
        CEDAR_EXECUTABLE,
        "-defect",
        str(REFERENCE_FILE),
        "--fin",
        str(dump_file)
    ]

    if dry_run:
        print(f"[DRY RUN] Would run in {output_dir}:")
        print(f"  {' '.join(cmd)}")
        return True

    # Run cedar in the output directory
    try:
        # When show_output is enabled, inherit stdout/stderr so progress is visible.
        # When not showing output, redirect stdout to DEVNULL to avoid pipe buffer deadlock
        # (cedar can produce large output), but capture stderr for error reporting.
        result = subprocess.run(
            cmd,
            cwd=str(output_dir),
            capture_output=not show_output,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"ERROR: Cedar failed for N={n}, M={m}", file=sys.stderr)
            if not show_output and result.stderr:
                print(f"  STDERR: {result.stderr}", file=sys.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"ERROR: Cedar timed out for N={n}, M={m}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Exception processing N={n}, M={m}: {e}", file=sys.stderr)
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

    if args.parallel > 1:
        # Parallel processing
        from multiprocessing import Pool

        tasks = [
            (n, m, args.dry_run, args.show_output)
            for n in range(args.n_start, args.n_end + 1)
            for m in range(args.m_start, args.m_end + 1)
        ]

        with Pool(processes=args.parallel) as pool:
            results = pool.starmap(process_cascade, tasks)

        for result in results:
            processed += 1
            if result is True:
                success += 1
            elif result is False:
                failed += 1
            else:
                skipped += 1
    else:
        # Sequential processing
        for n in range(args.n_start, args.n_end + 1):
            for m in range(args.m_start, args.m_end + 1):
                result = process_cascade(n, m, args.dry_run, args.show_output)

                # Progress update every 10 simulations
                processed += 1
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"Progress: {processed}/{total} ({100*processed/total:.1f}%) "
                        f"- Success: {success}, Failed: {failed}, Skipped: {skipped} "
                        f"- Rate: {rate:.1f} sim/min, ETA: {eta:.1f} min")

                if result is True:
                    success += 1
                elif result is False:
                    failed += 1
                else:
                    skipped += 1

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
