#!/usr/bin/env python3
"""
Cedar-based defect analysis for LAMMPS cascade simulations.

Wraps the proprietary ``cedar -defect`` command to identify point defects
(vacancies and self-interstitial atoms) in radiation damage cascade
simulations.  Cedar writes ``vac.dump``, ``sia.dump``, and ``sia_site.dump``
files in the specified output directory.

Usage::

    # Basic usage
    python cedar_defect_analysis.py dump.reference dump.end.min

    # With output directory
    python cedar_defect_analysis.py dump.reference dump.end.min --output-dir ./defects

    # Custom cedar executable path
    python cedar_defect_analysis.py dump.reference dump.end.min \\
        --cedar-executable /path/to/cedar
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default cedar executable path
DEFAULT_CEDAR_EXECUTABLE = "/flare/Cascaide/romano/software/cedar_2024jan/cedar"


def check_cedar_prerequisites(cedar_executable):
    """
    Check if the cedar executable exists and is runnable.

    Parameters
    ----------
    cedar_executable : str or Path
        Path to the cedar binary.

    Returns
    -------
    bool
        True if prerequisites are satisfied.
    """
    cedar_path = Path(cedar_executable)

    if not cedar_path.exists():
        logger.error("Cedar executable not found at %s", cedar_executable)
        return False

    if not cedar_path.is_file() or not os.access(cedar_path, os.X_OK):
        logger.error("Cedar path is not an executable file: %s", cedar_executable)
        return False

    return True


def cedar_defect_analysis(
    reference_dump,
    displaced_dump,
    output_dir=None,
    show_output=False,
    cedar_executable=None,
):
    """
    Run cedar defect analysis on a displaced dump file.

    Parameters
    ----------
    reference_dump : str or Path
        Path to the reference lattice dump file.
    displaced_dump : str or Path
        Path to the displaced dump file to analyze.
    output_dir : str or Path, optional
        Directory where cedar output files will be written
        (default: current directory).
    show_output : bool
        If True, stream cedar stdout/stderr to the console.
    cedar_executable : str or Path, optional
        Path to the cedar binary (default: ``DEFAULT_CEDAR_EXECUTABLE``).

    Returns
    -------
    dict
        ``"success"`` : bool — whether cedar completed successfully.
        ``"n_vacancies"`` : None — not parsed from cedar output.
        ``"n_sias"`` : None — not parsed from cedar output.
        ``"n_sia_sites"`` : None — not parsed from cedar output.
        ``"error"`` : str or None — error message if unsuccessful.
    """
    if cedar_executable is None:
        cedar_executable = DEFAULT_CEDAR_EXECUTABLE

    reference_dump = Path(reference_dump)
    displaced_dump = Path(displaced_dump)

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    # Check if input files exist
    if not reference_dump.exists():
        msg = f"Reference dump not found: {reference_dump}"
        logger.error(msg)
        return {"success": False, "n_vacancies": None, "n_sias": None,
                "n_sia_sites": None, "error": msg}

    if not displaced_dump.exists():
        msg = f"Displaced dump not found: {displaced_dump}"
        logger.warning(msg)
        return {"success": False, "n_vacancies": None, "n_sias": None,
                "n_sia_sites": None, "error": msg}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cedar command
    cmd = [
        str(cedar_executable),
        "-defect",
        str(reference_dump),
        "--fin",
        str(displaced_dump),
    ]

    logger.info("Running: %s", " ".join(cmd))
    logger.info("Output dir: %s", output_dir)

    # Run cedar in the output directory
    try:
        result = subprocess.run(
            cmd,
            cwd=str(output_dir),
            capture_output=not show_output,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            msg = f"Cedar returned non-zero exit code for {displaced_dump}"
            logger.error(msg)
            if not show_output and result.stderr:
                logger.error("  STDERR: %s", result.stderr)
                msg += f" | stderr: {result.stderr.strip()}"
            return {"success": False, "n_vacancies": None, "n_sias": None,
                    "n_sia_sites": None, "error": msg}

        return {"success": True, "n_vacancies": None, "n_sias": None,
                "n_sia_sites": None, "error": None}

    except subprocess.TimeoutExpired:
        msg = f"Cedar timed out for {displaced_dump}"
        logger.error(msg)
        return {"success": False, "n_vacancies": None, "n_sias": None,
                "n_sia_sites": None, "error": msg}
    except Exception as e:
        msg = f"Exception processing {displaced_dump}: {e}"
        logger.error(msg)
        return {"success": False, "n_vacancies": None, "n_sias": None,
                "n_sia_sites": None, "error": msg}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run cedar defect analysis on a LAMMPS dump file pair.  "
            "Identifies vacancies and SIAs by comparing a displaced "
            "configuration against a reference lattice."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s dump.reference dump.end.min\n"
            "  %(prog)s dump.reference dump.end.min --output-dir ./defects\n"
            "  %(prog)s dump.reference dump.end.min "
            "--cedar-executable /path/to/cedar\n"
        ),
    )

    parser.add_argument(
        "reference",
        help="Path to reference LAMMPS dump file (perfect lattice).",
    )
    parser.add_argument(
        "displaced",
        help="Path to displaced LAMMPS dump file (after cascade).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--cedar-executable",
        default=DEFAULT_CEDAR_EXECUTABLE,
        help=f"Path to cedar binary (default: {DEFAULT_CEDAR_EXECUTABLE}).",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Stream cedar stdout/stderr to the console.",
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

    # Validate cedar executable
    if not check_cedar_prerequisites(args.cedar_executable):
        return 1

    # Validate inputs
    ref_path = Path(args.reference)
    disp_path = Path(args.displaced)

    if not ref_path.exists():
        logger.error("Reference file not found: %s", ref_path)
        return 1
    if not disp_path.exists():
        logger.error("Displaced file not found: %s", disp_path)
        return 1

    # Run analysis
    result = cedar_defect_analysis(
        reference_dump=str(ref_path),
        displaced_dump=str(disp_path),
        output_dir=args.output_dir,
        show_output=args.show_output,
        cedar_executable=args.cedar_executable,
    )

    if result["success"]:
        print("Cedar defect analysis completed successfully.")
        print(f"Output files written to: {args.output_dir}")
    else:
        print(f"Cedar defect analysis FAILED: {result['error']}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
