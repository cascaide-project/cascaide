#!/usr/bin/env python3
"""
Batch defect analysis for cascade simulations.

Processes cascade configurations (indexed by N and M) and runs defect analysis
using either the OVITO or Cedar backend.  Each cascade has two variants
analyzed: minimized (``dump.end.min``) and unminimized (``dump.end``).

The default configuration covers 1000 cascades (N=1–100, M=1–10) from the
W_Chen19_Jan2026/sc80/50keV/pka1 dataset.

Usage::

    # Run with OVITO backend (default)
    python analyze_defects.py

    # Run with Cedar backend
    python analyze_defects.py --backend cedar

    # Process a subset with 4 parallel workers and write summary CSV
    python analyze_defects.py --n-end 10 --m-end 5 --parallel 4 \\
        --summary-csv results.csv

    # Preview what would be done
    python analyze_defects.py --dry-run
"""

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths (can be overridden via CLI arguments)
DEFAULT_CEDAR_EXECUTABLE = "/flare/Cascaide/romano/software/cedar_2024jan/cedar"
DEFAULT_REFERENCE_FILE = (
    "/flare/Cascaide/knight/lammps/W_Chen19_Jan2026/sc80/therm/dump.ref"
)
DEFAULT_BASE_DUMP_DIR = (
    "/flare/Cascaide/knight/lammps/W_Chen19_Jan2026/sc80/50keV/pka1"
)
DEFAULT_OUTPUT_BASE_DIR = (
    "/flare/Cascaide/romano/defect_analysis/sc80_50keV_pka1"
)


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------
# Each adapter takes the same arguments and returns a dict with at minimum:
#   "success": bool
#   "n_vacancies": int or None
#   "n_sias": int or None
#   "n_sia_sites": int or None
#   "error": str or None


def _run_ovito(reference, displaced, output_dir):
    """Run OVITO Wigner-Seitz defect analysis on a single cascade."""
    from ovito_defect_analysis import analyze_defects

    displaced = Path(displaced)
    if not displaced.exists():
        return {
            "success": False,
            "n_vacancies": None,
            "n_sias": None,
            "n_sia_sites": None,
            "error": f"Displaced dump not found: {displaced}",
        }

    try:
        result = analyze_defects(
            reference_file=str(reference),
            displaced_file=str(displaced),
            output_dir=str(output_dir),
            write_dump_files=True,
            return_data=False,
        )
        return {
            "success": True,
            "n_vacancies": result["n_vacancies"],
            "n_sias": result["n_sias"],
            "n_sia_sites": result["n_sia_sites"],
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "n_vacancies": None,
            "n_sias": None,
            "n_sia_sites": None,
            "error": str(e),
        }


def _run_cedar(reference, displaced, output_dir, cedar_executable=None):
    """Run Cedar defect analysis on a single cascade."""
    from cedar_defect_analysis import cedar_defect_analysis

    return cedar_defect_analysis(
        reference_dump=str(reference),
        displaced_dump=str(displaced),
        output_dir=str(output_dir),
        cedar_executable=cedar_executable,
    )


# ---------------------------------------------------------------------------
# Worker functions (top-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _worker_ovito(reference, displaced, output_dir):
    """Top-level worker for OVITO backend (must be picklable)."""
    return _run_ovito(reference, displaced, output_dir)


def _worker_cedar(reference, displaced, output_dir, cedar_executable):
    """Top-level worker for Cedar backend (must be picklable)."""
    return _run_cedar(reference, displaced, output_dir, cedar_executable)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def _build_tasks(args):
    """
    Generate the list of (n, m, state, displaced_path, output_dir) tasks.

    Returns
    -------
    list of dict
        Each dict has: n, m, state, displaced, output_dir.
    """
    tasks = []
    for n in range(args.n_start, args.n_end + 1):
        for m in range(args.m_start, args.m_end + 1):
            base_dir = Path(args.base_dir) / str(n) / str(m)
            output_base = Path(args.output_dir) / str(n) / str(m)

            # Minimized
            tasks.append({
                "n": n,
                "m": m,
                "state": "minimized",
                "displaced": str(base_dir / "dump.end.min"),
                "output_dir": str(output_base / "minimized"),
            })

            # Unminimized
            tasks.append({
                "n": n,
                "m": m,
                "state": "unminimized",
                "displaced": str(base_dir / "dump.end"),
                "output_dir": str(output_base / "unminimized"),
            })

    return tasks


def _should_skip(task, skip_existing):
    """Check whether this task's output already exists."""
    if not skip_existing:
        return False
    out_dir = Path(task["output_dir"])
    return (out_dir / "vac.dump").exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    """Main entry point for batch defect analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch defect analysis for cascade simulations.  "
            "Supports OVITO (open-source) and Cedar backends."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --backend ovito --parallel 4\n"
            "  %(prog)s --backend cedar --n-end 10 --m-end 5\n"
            "  %(prog)s --dry-run\n"
            "  %(prog)s --summary-csv results.csv --skip-existing\n"
        ),
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=["ovito", "cedar"],
        default="ovito",
        help="Defect analysis backend (default: ovito).",
    )

    # Paths
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE_FILE,
        help=f"Path to reference dump file (default: {DEFAULT_REFERENCE_FILE}).",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DUMP_DIR,
        help=f"Root of the cascade dump directory tree (default: {DEFAULT_BASE_DUMP_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_BASE_DIR,
        help=f"Root of the output directory tree (default: {DEFAULT_OUTPUT_BASE_DIR}).",
    )
    parser.add_argument(
        "--cedar-executable",
        default=DEFAULT_CEDAR_EXECUTABLE,
        help=(
            f"Path to cedar binary (default: {DEFAULT_CEDAR_EXECUTABLE}). "
            "Only used with --backend cedar."
        ),
    )

    # Range of configurations
    parser.add_argument(
        "--n-start",
        type=int,
        default=1,
        help="Starting N value (default: 1).",
    )
    parser.add_argument(
        "--n-end",
        type=int,
        default=100,
        help="Ending N value (default: 100).",
    )
    parser.add_argument(
        "--m-start",
        type=int,
        default=1,
        help="Starting M value (default: 1).",
    )
    parser.add_argument(
        "--m-end",
        type=int,
        default=10,
        help="Ending M value (default: 10).",
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cascades whose output directory already has vac.dump.",
    )

    # Output options
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Write per-cascade defect counts to this CSV file.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Stream backend stdout/stderr (Cedar only).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate cedar prerequisites if needed
    if args.backend == "cedar" and not args.dry_run:
        from cedar_defect_analysis import check_cedar_prerequisites
        if not check_cedar_prerequisites(args.cedar_executable):
            return 1

    # Validate reference file
    if not args.dry_run:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            logger.error("Reference file not found: %s", ref_path)
            return 1

    # Build task list
    tasks = _build_tasks(args)
    total = len(tasks)

    # Statistics
    processed = 0
    success = 0
    failed = 0
    skipped = 0
    results_log = []  # For summary CSV

    start_time = datetime.now()

    print(f"Starting batch defect analysis at {start_time}")
    print(f"Backend: {args.backend}")
    print(f"Configurations: {total} tasks "
          f"(N={args.n_start}-{args.n_end}, M={args.m_start}-{args.m_end}, "
          f"x2 states)")
    print(f"Reference: {args.reference}")
    print(f"Input dir: {args.base_dir}")
    print(f"Output dir: {args.output_dir}")
    if args.backend == "cedar":
        print(f"Cedar executable: {args.cedar_executable}")
    print(f"Parallel workers: {args.parallel}")
    if args.skip_existing:
        print("Skipping existing results")
    print("-" * 80)

    if args.dry_run:
        for task in tasks:
            skip = _should_skip(task, args.skip_existing)
            status = "[SKIP]" if skip else "[DRY RUN]"
            print(f"  {status} N={task['n']}, M={task['m']}, "
                  f"state={task['state']}")
            print(f"    Displaced: {task['displaced']}")
            print(f"    Output:    {task['output_dir']}")
            if skip:
                skipped += 1
            else:
                success += 1
            processed += 1
        print(f"\n{processed} tasks total, {skipped} would be skipped")
        return 0

    # Select executor type based on backend:
    #   - OVITO: ProcessPoolExecutor (required per OVITO docs)
    #   - Cedar: ThreadPoolExecutor  (subprocess-based, threads are fine)
    if args.backend == "ovito":
        ExecutorClass = ProcessPoolExecutor
    else:
        ExecutorClass = ThreadPoolExecutor

    # Submit and process tasks
    with ExecutorClass(max_workers=args.parallel) as executor:
        future_to_task = {}

        for task in tasks:
            # Check skip
            if _should_skip(task, args.skip_existing):
                skipped += 1
                processed += 1
                results_log.append({
                    "n": task["n"],
                    "m": task["m"],
                    "state": task["state"],
                    "n_vacancies": "",
                    "n_sias": "",
                    "n_sia_sites": "",
                    "success": "skipped",
                    "error": "",
                })
                continue

            # Submit to executor
            if args.backend == "ovito":
                future = executor.submit(
                    _worker_ovito,
                    args.reference,
                    task["displaced"],
                    task["output_dir"],
                )
            else:
                future = executor.submit(
                    _worker_cedar,
                    args.reference,
                    task["displaced"],
                    task["output_dir"],
                    args.cedar_executable,
                )

            future_to_task[future] = task

        # Collect results
        submitted = len(future_to_task)
        completed = 0

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            processed += 1

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "success": False,
                    "n_vacancies": None,
                    "n_sias": None,
                    "n_sia_sites": None,
                    "error": f"Worker exception: {e}",
                }

            if result["success"]:
                success += 1
            else:
                failed += 1

            # Log for CSV
            results_log.append({
                "n": task["n"],
                "m": task["m"],
                "state": task["state"],
                "n_vacancies": result.get("n_vacancies", ""),
                "n_sias": result.get("n_sias", ""),
                "n_sia_sites": result.get("n_sia_sites", ""),
                "success": result["success"],
                "error": result.get("error", "") or "",
            })

            # Progress
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = submitted - completed
            eta = remaining / rate if rate > 0 else 0

            status_parts = [
                f"Progress: {processed}/{total} ({100 * processed / total:.1f}%)",
                f"Success: {success}",
                f"Failed: {failed}",
            ]
            if skipped > 0:
                status_parts.append(f"Skipped: {skipped}")
            status_parts.append(f"Rate: {rate:.1f}/min")
            status_parts.append(f"ETA: {eta:.1f} min")

            if result["success"]:
                detail = ""
                if result.get("n_vacancies") is not None:
                    detail = (
                        f" [vac={result['n_vacancies']}, "
                        f"sia={result['n_sias']}]"
                    )
                print(
                    f"  OK  N={task['n']:>3d} M={task['m']:>2d} "
                    f"{task['state']:<12s}{detail}  |  "
                    + " - ".join(status_parts)
                )
            else:
                print(
                    f"  FAIL N={task['n']:>3d} M={task['m']:>2d} "
                    f"{task['state']:<12s}  |  "
                    + " - ".join(status_parts)
                )
                logger.warning(
                    "  Error: %s", result.get("error", "unknown")
                )

    # Write summary CSV
    if args.summary_csv and results_log:
        csv_path = Path(args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort by n, m, state for consistent output
        results_log.sort(key=lambda r: (r["n"], r["m"], r["state"]))

        fieldnames = [
            "n", "m", "state", "n_vacancies", "n_sias", "n_sia_sites",
            "success", "error",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_log)

        print(f"\nSummary CSV written to: {csv_path}")

    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("-" * 80)
    print(f"Analysis complete at {end_time}")
    print(f"Total time: {duration / 60:.1f} minutes")
    print(f"Processed: {processed}/{total}")
    print(f"  Success:  {success}")
    print(f"  Failed:   {failed}")
    print(f"  Skipped:  {skipped}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
