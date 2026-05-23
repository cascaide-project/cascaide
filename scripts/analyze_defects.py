#!/usr/bin/env python3
"""
Batch defect analysis for cascade simulations.

Processes one or more cascade datasets and runs defect analysis using either
the OVITO or Cedar backend.  Each cascade has two variants analyzed: minimized
(``dump.end.min`` / ``dump.end.min.gz``) and unminimized
(``dump.end`` / ``dump.end.gz``).

Datasets can be specified via a YAML configuration file (recommended for
multi-dataset campaigns) **or** via legacy CLI flags for a single dataset.

Usage::

    # Multi-dataset mode (recommended)
    python analyze_defects.py --config datasets.yaml

    # Preview what would be done
    python analyze_defects.py --config datasets.yaml --dry-run

    # Single-dataset legacy mode (backward-compatible)
    python analyze_defects.py --reference dump.ref --base-dir ./pka1

    # Override config-level settings from CLI
    python analyze_defects.py --config datasets.yaml --backend cedar --parallel 8
"""

import argparse
import csv
import json
import logging
import math
import os
import re
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
# Cascade discovery
# ---------------------------------------------------------------------------


def discover_cascades(base_dir):
    """
    Recursively search *base_dir* for cascade dump files.

    A cascade is any directory that contains ``dump.end`` or ``dump.end.gz``.
    If the same directory also has ``dump.end.min`` or ``dump.end.min.gz``,
    the minimized variant is recorded too.  Plain files take precedence over
    their gzipped counterparts when both are present.

    Cascades are sorted by their path relative to *base_dir* (using
    numeric sorting on individual path components where possible) and
    assigned sequential 1-based integer IDs.

    Parameters
    ----------
    base_dir : str or Path
        Root of the cascade dump tree (e.g. ``…/50keV/pka1``).

    Returns
    -------
    list of dict
        Each dict has:
        - ``cascade_id``    : int – 1-based sequential identifier.
        - ``rel_path``      : str – path components relative to base_dir
                              (e.g. ``"3/7"``).
        - ``displaced``     : str – absolute path to ``dump.end`` (or
                              ``dump.end.gz``).
        - ``displaced_min`` : str or None – absolute path to the minimized
                              dump (``dump.end.min`` / ``dump.end.min.gz``,
                              or None if absent).
        - ``log_file``      : str or None – absolute path to
                              ``log.lammps`` in the cascade directory
                              (None if absent).
        - ``pka_energy_eV`` : float or None – PKA kinetic energy in eV
                              parsed from ``log.lammps`` (None if the log
                              is absent or the value cannot be found).
        - ``pka_id``        : int or None – LAMMPS atom ID of the PKA
                              parsed from ``log.lammps`` (None if absent).
        - ``path_parts``    : list of str – individual directory names
                              (e.g. ``["3", "7"]``).
    """
    base = Path(base_dir).resolve()
    if not base.is_dir():
        logger.error("Base directory does not exist: %s", base)
        return []

    # Collect cascade directories from both plain and gzipped dump files,
    # preferring the plain variant when both exist in the same directory.
    seen_dirs = {}
    for pattern in ("dump.end", "dump.end.gz"):
        for dump_end in base.rglob(pattern):
            cascade_dir = dump_end.parent
            if cascade_dir not in seen_dirs:
                seen_dirs[cascade_dir] = dump_end

    cascades = []
    for cascade_dir, dump_end in seen_dirs.items():
        rel = cascade_dir.relative_to(base)
        parts = list(rel.parts)

        # Prefer plain dump.end.min; fall back to gzipped variant.
        displaced_min = cascade_dir / "dump.end.min"
        if not displaced_min.exists():
            displaced_min_gz = cascade_dir / "dump.end.min.gz"
            displaced_min = displaced_min_gz if displaced_min_gz.exists() else None

        log_file = cascade_dir / "log.lammps"
        pka_energy_eV = None
        pka_id = None
        if log_file.exists():
            pka_id, pka_energy_eV = parse_pka_info_from_log(log_file)
        else:
            log_file = None

        cascades.append({
            "rel_path": str(rel),
            "displaced": str(dump_end),
            "displaced_min": str(displaced_min) if displaced_min is not None else None,
            "log_file": str(log_file) if log_file is not None else None,
            "pka_energy_eV": pka_energy_eV,
            "pka_id": pka_id,
            "path_parts": parts,
        })

    # Sort using numeric-aware key so "2" < "10"
    def _sort_key(c):
        parts = []
        for p in c["path_parts"]:
            try:
                parts.append((0, int(p)))
            except ValueError:
                parts.append((1, p))
        return parts

    cascades.sort(key=_sort_key)

    # Assign deterministic IDs
    for idx, c in enumerate(cascades, start=1):
        c["cascade_id"] = idx

    logger.info(
        "Discovered %d cascades in %s", len(cascades), base_dir
    )
    return cascades


def _id_format(total):
    """Return a zero-padded format string for cascade IDs."""
    width = max(4, len(str(total)))
    return f"{{:0{width}d}}"


# ---------------------------------------------------------------------------
# LAMMPS input parser
# ---------------------------------------------------------------------------


def parse_lammps_inputs(base_dir):
    """
    Search for LAMMPS input files near *base_dir* and extract metadata.

    Looks for ``in.cascade`` in *base_dir* and ``in.therm`` in sibling /
    grandparent ``therm`` or ``therm_2`` directories.  Falls back to any
    ``in.*`` file found nearby.

    Parameters
    ----------
    base_dir : str or Path
        Root of the cascade dump tree (the per-energy/pka directory).

    Returns
    -------
    dict
        Extracted metadata (keys with ``None`` values if not found):

        - ``material`` : str  (from pair_coeff element symbols)
        - ``potential`` : str  (interatomic potential filename)
        - ``pair_style`` : str
        - ``temperature_K`` : float
        - ``pka_energy_eV`` : float
        - ``timestep_ps`` : float
        - ``lattice_constant_angstrom`` : float
        - ``cascade_duration_ps`` : float
        - ``units`` : str
        - ``pka_id`` : int
        - ``n_therm_configs`` : int  (number of thermalized restart files)
        - ``therm_input_file`` : str  (path found)
        - ``cascade_input_file`` : str  (path found)
    """
    base = Path(base_dir).resolve()

    result = {
        "material": None,
        "potential": None,
        "pair_style": None,
        "temperature_K": None,
        "pka_energy_eV": None,
        "timestep_ps": None,
        "lattice_constant_angstrom": None,
        "cascade_duration_ps": None,
        "units": None,
        "pka_id": None,
        "n_therm_configs": None,
        "therm_input_file": None,
        "cascade_input_file": None,
    }

    # --- Locate input files ---
    cascade_input = base / "in.cascade"
    if not cascade_input.exists():
        # Try one level up
        cascade_input = base.parent / "in.cascade"
    if cascade_input.exists():
        result["cascade_input_file"] = str(cascade_input)
        _parse_cascade_input(cascade_input, result)

    # Search for therm input: look in sibling/grandparent therm dirs
    therm_input = None
    for candidate in [
        base.parent / "therm" / "in.therm",
        base.parent / "therm_2" / "in.therm",
        base.parent.parent / "therm" / "in.therm",
        base.parent.parent / "therm_2" / "in.therm",
    ]:
        if candidate.exists():
            therm_input = candidate
            break

    if therm_input is not None:
        result["therm_input_file"] = str(therm_input)
        _parse_therm_input(therm_input, result)

    return result


def _parse_cascade_input(filepath, result):
    """Parse an in.cascade file and update *result* dict in-place."""
    try:
        text = Path(filepath).read_text(errors="replace")
    except OSError as e:
        logger.warning("Could not read %s: %s", filepath, e)
        return

    # units
    m = re.search(r"^\s*units\s+(\S+)", text, re.MULTILINE)
    if m:
        result["units"] = m.group(1)

    # pair_style
    m = re.search(r"^\s*pair_style\s+(.+)", text, re.MULTILINE)
    if m:
        result["pair_style"] = m.group(1).strip()

    # pair_coeff → potential file and element symbols
    m = re.search(r"^\s*pair_coeff\s+\*\s+\*\s+(\S+)\s+(.*)", text, re.MULTILINE)
    if m:
        result["potential"] = Path(m.group(1)).name  # just filename
        elements = m.group(2).strip().split()
        if elements:
            result["material"] = " ".join(elements)

    # PKA energy: variable Ek equal 50*1000.0  →  50000 eV
    m = re.search(r"^\s*variable\s+Ek\s+equal\s+(.+)", text, re.MULTILINE)
    if m:
        try:
            result["pka_energy_eV"] = float(eval(m.group(1).strip()))
        except Exception:
            pass

    # PKA id
    m = re.search(r"^\s*variable\s+pkaid\s+equal\s+(\d+)", text, re.MULTILINE)
    if m:
        result["pka_id"] = int(m.group(1))

    # Cascade duration
    m = re.search(
        r"^\s*variable\s+duration\s+equal\s+(\S+)", text, re.MULTILINE
    )
    if m:
        try:
            result["cascade_duration_ps"] = float(m.group(1))
        except ValueError:
            pass

    # timestep
    m = re.search(r"^\s*timestep\s+(\S+)", text, re.MULTILINE)
    if m:
        try:
            result["timestep_ps"] = float(m.group(1))
        except ValueError:
            pass

    # Temperature from variable TEMP
    m = re.search(r"^\s*variable\s+TEMP\s+equal\s+(\S+)", text, re.MULTILINE)
    if m:
        try:
            result["temperature_K"] = float(m.group(1))
        except ValueError:
            pass

    # Lattice constant: variable a0 equal 63.30497/20
    m = re.search(r"^\s*variable\s+a0\s+equal\s+(.+)", text, re.MULTILINE)
    if m:
        try:
            result["lattice_constant_angstrom"] = float(eval(m.group(1).strip()))
        except Exception:
            pass


def _parse_therm_input(filepath, result):
    """Parse an in.therm file and update *result* dict in-place."""
    try:
        text = Path(filepath).read_text(errors="replace")
    except OSError as e:
        logger.warning("Could not read %s: %s", filepath, e)
        return

    # Temperature (may already be set by cascade input, therm is fallback)
    if result.get("temperature_K") is None:
        m = re.search(
            r"^\s*variable\s+TEMP\s+equal\s+(\S+)", text, re.MULTILINE
        )
        if m:
            try:
                result["temperature_K"] = float(m.group(1))
            except ValueError:
                pass

    # potential (fallback)
    if result.get("potential") is None:
        m = re.search(
            r"^\s*pair_coeff\s+\*\s+\*\s+(\S+)\s+(.*)", text, re.MULTILINE
        )
        if m:
            result["potential"] = Path(m.group(1)).name
            elements = m.group(2).strip().split()
            if elements and result.get("material") is None:
                result["material"] = " ".join(elements)

    # Number of thermalization configs: variable i loop 1 N
    m = re.search(r"^\s*variable\s+\w+\s+loop\s+\d+\s+(\d+)", text, re.MULTILINE)
    if m:
        result["n_therm_configs"] = int(m.group(1))


def parse_pka_info_from_log(log_path):
    """
    Extract the PKA atom ID and kinetic energy from a LAMMPS ``log.lammps``.

    LAMMPS logs both values via a ``print`` statement of the form::

        PKAid 45919 Energy 5793.36

    If that line is absent, the energy falls back to the last resolved
    ``variable Ek equal`` line, and the ID is returned as ``None``.

    Parameters
    ----------
    log_path : str or Path
        Path to the LAMMPS log file.

    Returns
    -------
    tuple of (int or None, float or None)
        ``(pka_id, pka_energy_eV)``.  Either value may be ``None`` if it
        cannot be found or the file is unreadable.
    """
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError as e:
        logger.warning("Could not read log file %s: %s", log_path, e)
        return None, None

    # Primary: explicit print line "PKAid <id> Energy <eV>"
    m = re.search(r"^PKAid\s+(\d+)\s+Energy\s+(\S+)", text, re.MULTILINE)
    if m:
        try:
            pka_id = int(m.group(1))
            pka_energy_eV = float(m.group(2))
            return pka_id, pka_energy_eV
        except ValueError:
            pass

    # Fallback: last resolved "variable Ek equal <expr>" line.
    # LAMMPS echoes both the template form (${my_pka}*1000.0) and the
    # expanded form (5.793360*1000.0); take the last match so we get the
    # numeric expression rather than the template.
    matches = re.findall(
        r"^\s*variable\s+Ek\s+equal\s+(.+)", text, re.MULTILINE
    )
    for expr in reversed(matches):
        expr = expr.strip()
        # Skip template expressions that still contain variable references
        if "$" in expr or "{" in expr:
            continue
        try:
            return None, float(eval(expr))  # noqa: S307 — log files are local/trusted
        except Exception:
            continue

    logger.warning("Could not find PKA energy in log file: %s", log_path)
    return None, None


def parse_pka_energy_from_log(log_path):
    """
    Extract the PKA kinetic energy (eV) from a LAMMPS ``log.lammps`` file.

    .. deprecated::
        Use :func:`parse_pka_info_from_log` to obtain both the PKA ID and
        energy in a single pass.
    """
    _, energy = parse_pka_info_from_log(log_path)
    return energy


# ---------------------------------------------------------------------------
# PKA position lookup from reference dump
# ---------------------------------------------------------------------------


def lookup_pka_positions(reference_path, pka_ids):
    """
    Look up Cartesian positions for a set of PKA atom IDs in a reference dump.

    Reads the reference dump file once and returns a mapping from atom ID to
    ``{"x": float, "y": float, "z": float}`` in Angstrom.  IDs not found in
    the dump are absent from the returned dict.

    The dump must contain an ``ITEM: ATOMS`` header with at minimum the
    columns ``id``, ``x``, ``y``, and ``z`` (in any order).

    Parameters
    ----------
    reference_path : str or Path
        Path to the LAMMPS reference dump file (plain text or gzipped).
    pka_ids : collection of int
        Atom IDs to look up.

    Returns
    -------
    dict
        ``{atom_id: {"x": float, "y": float, "z": float}}``
    """
    import gzip

    needed = set(pka_ids) - {None}
    if not needed:
        return {}

    ref = Path(reference_path)
    if not ref.exists():
        logger.warning("Reference dump not found for PKA lookup: %s", ref)
        return {}

    positions = {}
    try:
        opener = gzip.open if ref.suffix == ".gz" else open
        with opener(ref, "rt", errors="replace") as fh:
            col_id = col_x = col_y = col_z = None
            in_atoms = False
            for line in fh:
                if not in_atoms:
                    if line.startswith("ITEM: ATOMS"):
                        cols = line.split()[2:]  # strip "ITEM:" and "ATOMS"
                        try:
                            col_id = cols.index("id")
                            col_x = cols.index("x")
                            col_y = cols.index("y")
                            col_z = cols.index("z")
                        except ValueError:
                            logger.warning(
                                "Reference dump missing id/x/y/z columns: %s",
                                ref,
                            )
                            return {}
                        in_atoms = True
                    continue
                if line.startswith("ITEM:"):
                    # New section — stop (single-frame dump)
                    break
                parts = line.split()
                try:
                    aid = int(parts[col_id])
                except (IndexError, ValueError):
                    continue
                if aid in needed:
                    try:
                        positions[aid] = {
                            "x": float(parts[col_x]),
                            "y": float(parts[col_y]),
                            "z": float(parts[col_z]),
                        }
                    except (IndexError, ValueError):
                        pass
                    if len(positions) == len(needed):
                        break  # found everything, no need to read further
    except OSError as e:
        logger.warning("Could not read reference dump for PKA lookup: %s", e)

    missing = needed - set(positions)
    if missing:
        logger.warning(
            "PKA atom ID(s) not found in reference dump %s: %s",
            ref,
            sorted(missing),
        )
    return positions


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------


def load_config(config_path):
    """
    Load and validate a YAML dataset configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Full config with ``"datasets"`` list and optional global settings.

    Raises
    ------
    SystemExit
        On validation errors.
    """
    try:
        import yaml
    except ImportError:
        logger.error(
            "PyYAML is required for --config mode.  "
            "Install it with: python -m pip install pyyaml"
        )
        sys.exit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "datasets" not in config:
        logger.error("Config file must contain a 'datasets' key.")
        sys.exit(1)

    required_keys = {"name", "reference", "base_dir", "output_dir"}
    for i, ds in enumerate(config["datasets"]):
        missing = required_keys - set(ds.keys())
        if missing:
            logger.error(
                "Dataset #%d is missing required keys: %s", i + 1, missing
            )
            sys.exit(1)

    return config


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------
# Each adapter takes the same arguments and returns a dict with at minimum:
#   "success": bool
#   "n_vacancies": int or None
#   "n_sias": int or None
#   "n_sia_sites": int or None
#   "error": str or None


def _run_ovito(reference, displaced, output_dir, output_prefix=""):
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
            output_prefix=output_prefix,
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


def _run_cedar(reference, displaced, output_dir, output_prefix="",
               cedar_executable=None):
    """Run Cedar defect analysis on a single cascade.

    Cedar always writes ``vac.dump``, ``sia.dump``, ``sia_site.dump`` into
    its working directory.  After completion, the files are renamed to
    include the *output_prefix* so they sit in the flat output layout.
    """
    import shutil
    import tempfile

    from cedar_defect_analysis import cedar_defect_analysis

    # Run cedar in a temporary sub-directory to avoid collisions
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(output_dir)) as tmpdir:
        result = cedar_defect_analysis(
            reference_dump=str(reference),
            displaced_dump=str(displaced),
            output_dir=tmpdir,
            cedar_executable=cedar_executable,
        )

        if result["success"]:
            # Move and rename output files
            for base_name in ("vac.dump", "sia.dump", "sia_site.dump"):
                src = Path(tmpdir) / base_name
                if src.exists():
                    dst = output_dir / f"{output_prefix}{base_name}"
                    shutil.move(str(src), str(dst))

    return result


# ---------------------------------------------------------------------------
# Worker functions (top-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _worker_ovito(reference, displaced, output_dir, output_prefix):
    """Top-level worker for OVITO backend (must be picklable)."""
    return _run_ovito(reference, displaced, output_dir, output_prefix)


def _worker_cedar(reference, displaced, output_dir, output_prefix,
                  cedar_executable):
    """Top-level worker for Cedar backend (must be picklable)."""
    return _run_cedar(reference, displaced, output_dir, output_prefix,
                      cedar_executable)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def build_tasks(cascades, dataset_cfg, id_fmt=None):
    """
    Generate analysis tasks from discovered cascades.

    Parameters
    ----------
    cascades : list of dict
        Output of :func:`discover_cascades`.
    dataset_cfg : dict
        Dataset configuration dict (must have ``output_dir``, ``reference``).
    id_fmt : str or None
        Format string for cascade IDs (e.g. ``"{:04d}"``).  Computed
        automatically if *None*.

    Returns
    -------
    list of dict
        Each dict has: cascade_id, rel_path, state, displaced, output_dir,
        output_prefix.
    """
    if id_fmt is None:
        id_fmt = _id_format(len(cascades))
    output_dir = dataset_cfg["output_dir"]

    tasks = []
    for cascade in cascades:
        cid = cascade["cascade_id"]
        cid_str = id_fmt.format(cid)

        pka_energy_eV = cascade.get("pka_energy_eV")

        # Unminimized
        tasks.append({
            "cascade_id": cid,
            "rel_path": cascade["rel_path"],
            "state": "unminimized",
            "displaced": cascade["displaced"],
            "output_dir": output_dir,
            "output_prefix": f"{cid_str}_",
            "pka_energy_eV": pka_energy_eV,
        })

        # Minimized (if available)
        if cascade["displaced_min"] is not None:
            tasks.append({
                "cascade_id": cid,
                "rel_path": cascade["rel_path"],
                "state": "minimized",
                "displaced": cascade["displaced_min"],
                "output_dir": output_dir,
                "output_prefix": f"{cid_str}_min_",
                "pka_energy_eV": pka_energy_eV,
            })

    return tasks


def _read_dump_atom_count(filepath):
    """Return the atom count from a LAMMPS dump file without loading all data.

    Reads only the header until ``ITEM: NUMBER OF ATOMS`` is found, then
    returns the integer on the following line.  Returns ``None`` if the file
    does not exist or cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            for line in f:
                if line.strip() == "ITEM: NUMBER OF ATOMS":
                    count_line = next(f, None)
                    if count_line is not None:
                        return int(count_line.strip())
    except (OSError, ValueError):
        pass
    return None


def _should_skip(task, skip_existing):
    """Check whether this task's output already exists."""
    if not skip_existing:
        return False
    out_dir = Path(task["output_dir"])
    return (out_dir / f"{task['output_prefix']}vac.dump").exists()


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def run_tasks(tasks, backend, parallel, reference, cedar_executable=None,
              skip_existing=False):
    """
    Execute a list of analysis tasks and return results.

    Parameters
    ----------
    tasks : list of dict
        Tasks from :func:`build_tasks`.
    backend : str
        ``"ovito"`` or ``"cedar"``.
    parallel : int
        Number of workers.
    reference : str
        Path to reference dump file.
    cedar_executable : str or None
        Path to cedar binary (Cedar backend only).
    skip_existing : bool
        Whether to skip tasks whose output already exists.

    Returns
    -------
    list of dict
        One result dict per task with keys: cascade_id, rel_path, state,
        n_vacancies, n_sias, n_sia_sites, success, error, output_files.
    """
    total = len(tasks)
    processed = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0
    results_log = []

    start_time = datetime.now()

    if backend == "ovito":
        ExecutorClass = ProcessPoolExecutor
    else:
        ExecutorClass = ThreadPoolExecutor

    with ExecutorClass(max_workers=parallel) as executor:
        future_to_task = {}

        for task in tasks:
            if _should_skip(task, skip_existing):
                skipped_count += 1
                processed += 1
                prefix = task["output_prefix"]
                out_dir = Path(task["output_dir"])
                n_vacancies = _read_dump_atom_count(out_dir / f"{prefix}vac.dump")
                n_sias = _read_dump_atom_count(out_dir / f"{prefix}sia.dump")
                n_sia_sites = _read_dump_atom_count(
                    out_dir / f"{prefix}sia_site.dump"
                )
                counts_ok = n_vacancies is not None
                if counts_ok:
                    success_count += 1
                results_log.append({
                    "cascade_id": task["cascade_id"],
                    "rel_path": task["rel_path"],
                    "state": task["state"],
                    "pka_energy_eV": task.get("pka_energy_eV"),
                    "n_vacancies": n_vacancies,
                    "n_sias": n_sias,
                    "n_sia_sites": n_sia_sites,
                    "success": True if counts_ok else "skipped",
                    "error": None if counts_ok else "",
                    "output_files": [
                        f"{prefix}{name}"
                        for name in ("vac.dump", "sia.dump", "sia_site.dump")
                    ] if counts_ok else [],
                })
                cid = task["cascade_id"]
                if counts_ok:
                    print(
                        f"  SKIP #{cid:>4d} ({task['rel_path']}) "
                        f"{task['state']:<12s}"
                        f" [vac={n_vacancies}, sia={n_sias}]"
                    )
                continue

            if backend == "ovito":
                future = executor.submit(
                    _worker_ovito,
                    reference,
                    task["displaced"],
                    task["output_dir"],
                    task["output_prefix"],
                )
            else:
                future = executor.submit(
                    _worker_cedar,
                    reference,
                    task["displaced"],
                    task["output_dir"],
                    task["output_prefix"],
                    cedar_executable,
                )

            future_to_task[future] = task

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
                success_count += 1
            else:
                failed_count += 1

            # Build output file list
            prefix = task["output_prefix"]
            output_files = []
            if result["success"]:
                for name in ("vac.dump", "sia.dump", "sia_site.dump"):
                    output_files.append(f"{prefix}{name}")

            results_log.append({
                "cascade_id": task["cascade_id"],
                "rel_path": task["rel_path"],
                "state": task["state"],
                "pka_energy_eV": task.get("pka_energy_eV"),
                "n_vacancies": result.get("n_vacancies"),
                "n_sias": result.get("n_sias"),
                "n_sia_sites": result.get("n_sia_sites"),
                "success": result["success"],
                "error": result.get("error", "") or "",
                "output_files": output_files,
            })

            # Progress
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = submitted - completed
            eta = remaining / rate if rate > 0 else 0

            status_parts = [
                f"Progress: {processed}/{total} "
                f"({100 * processed / total:.1f}%)",
                f"OK: {success_count}",
                f"Fail: {failed_count}",
            ]
            if skipped_count > 0:
                status_parts.append(f"Skip: {skipped_count}")
            status_parts.append(f"Rate: {rate:.1f}/min")
            status_parts.append(f"ETA: {eta:.1f} min")

            cid = task["cascade_id"]
            if result["success"]:
                detail = ""
                if result.get("n_vacancies") is not None:
                    detail = (
                        f" [vac={result['n_vacancies']}, "
                        f"sia={result['n_sias']}]"
                    )
                print(
                    f"  OK   #{cid:>4d} ({task['rel_path']}) "
                    f"{task['state']:<12s}{detail}  |  "
                    + " - ".join(status_parts)
                )
            else:
                print(
                    f"  FAIL #{cid:>4d} ({task['rel_path']}) "
                    f"{task['state']:<12s}  |  "
                    + " - ".join(status_parts)
                )
                logger.warning(
                    "  Error: %s", result.get("error", "unknown")
                )

    return results_log


# ---------------------------------------------------------------------------
# Metadata writer
# ---------------------------------------------------------------------------


def write_metadata(output_dir, dataset_cfg, sim_metadata, cascades,
                   results_log, backend, start_time, end_time):
    """
    Write ``metadata.json`` summarising a dataset's analysis.

    Parameters
    ----------
    output_dir : str or Path
        Dataset output directory.
    dataset_cfg : dict
        Original dataset configuration.
    sim_metadata : dict
        Parsed LAMMPS metadata from :func:`parse_lammps_inputs`.
    cascades : list of dict
        Discovered cascades from :func:`discover_cascades`.
    results_log : list of dict
        Per-task results from :func:`run_tasks`.
    backend : str
        Backend name.
    start_time, end_time : datetime
        Wall-clock times.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group results by cascade_id
    results_by_cascade = {}
    for r in results_log:
        cid = r["cascade_id"]
        results_by_cascade.setdefault(cid, {})[r["state"]] = r

    cascade_records = []
    for cascade in cascades:
        cid = cascade["cascade_id"]
        states = {}
        for state_name in ("minimized", "unminimized"):
            r = results_by_cascade.get(cid, {}).get(state_name)
            if r is not None:
                states[state_name] = {
                    "source_file": Path(
                        cascade.get(
                            "displaced_min" if state_name == "minimized"
                            else "displaced", ""
                        ) or ""
                    ).name,
                    "n_vacancies": r.get("n_vacancies"),
                    "n_sias": r.get("n_sias"),
                    "n_sia_sites": r.get("n_sia_sites"),
                    "success": r["success"],
                    "error": r.get("error", "") or None,
                    "output_files": r.get("output_files", []),
                }
        cascade_records.append({
            "cascade_id": cid,
            "source_path": cascade["rel_path"],
            "pka_energy_eV": cascade.get("pka_energy_eV"),
            "pka_id": cascade.get("pka_id"),
            "pka_position_angstrom": cascade.get("pka_position_angstrom"),
            "log_file": cascade.get("log_file"),
            "states": states,
        })

    # Counts
    ok = sum(1 for r in results_log if r["success"] is True)
    fail = sum(1 for r in results_log if r["success"] is False)
    skip = sum(1 for r in results_log if r["success"] == "skipped")

    # Merge explicit overrides into sim_metadata
    sim_info = {k: v for k, v in sim_metadata.items() if v is not None}
    for key in ("pka_energy_keV", "supercell", "material",
                "temperature_K", "potential"):
        if key in dataset_cfg and dataset_cfg[key] is not None:
            sim_info[key] = dataset_cfg[key]

    metadata = {
        "dataset_name": dataset_cfg["name"],
        "created": end_time.isoformat(),
        "backend": backend,
        "reference_file": str(dataset_cfg["reference"]),
        "base_dir": str(dataset_cfg["base_dir"]),
        "simulation_metadata": sim_info,
        "n_cascades": len(cascades),
        "cascades": cascade_records,
        "summary": {
            "total_tasks": len(results_log),
            "successful": ok,
            "failed": fail,
            "skipped": skip,
            "wall_time_seconds": (end_time - start_time).total_seconds(),
        },
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Metadata written to %s", meta_path)
    return meta_path


# ---------------------------------------------------------------------------
# Summary CSV (optional, backward-compatible)
# ---------------------------------------------------------------------------


def write_summary_csv(csv_path, all_results):
    """Write a combined summary CSV from results across all datasets."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_results.sort(
        key=lambda r: (r.get("dataset", ""), r["cascade_id"], r["state"])
    )

    fieldnames = [
        "dataset", "cascade_id", "rel_path", "state", "pka_energy_eV",
        "n_vacancies", "n_sias", "n_sia_sites", "success", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSummary CSV written to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    """Main entry point for batch defect analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch defect analysis for cascade simulations.  "
            "Supports OVITO (open-source) and Cedar backends.  "
            "Use --config with a YAML file for multi-dataset campaigns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Multi-dataset mode\n"
            "  %(prog)s --config datasets.yaml\n"
            "  %(prog)s --config datasets.yaml --backend cedar --parallel 8\n"
            "\n"
            "  # Single-dataset legacy mode\n"
            "  %(prog)s --reference dump.ref --base-dir ./pka1 "
            "--output-dir ./out\n"
            "\n"
            "  # Preview\n"
            "  %(prog)s --config datasets.yaml --dry-run\n"
        ),
    )

    # Config file
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to YAML config listing datasets.  When provided, "
            "--reference/--base-dir/--output-dir are ignored."
        ),
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=["ovito", "cedar"],
        default=None,
        help="Defect analysis backend (default: ovito).",
    )

    # Paths (legacy single-dataset mode)
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE_FILE,
        help="Path to reference dump file (single-dataset mode).",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DUMP_DIR,
        help="Root of cascade dump directory tree (single-dataset mode).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_BASE_DIR,
        help="Root of output directory tree (single-dataset mode).",
    )
    parser.add_argument(
        "--cedar-executable",
        default=DEFAULT_CEDAR_EXECUTABLE,
        help=(
            f"Path to cedar binary (default: {DEFAULT_CEDAR_EXECUTABLE}). "
            "Only used with --backend cedar."
        ),
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
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
        default=None,
        help="Skip cascades whose output already exists.",
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

    # ---------------------------------------------------------------
    # Build dataset list (from config or from CLI flags)
    # ---------------------------------------------------------------
    if args.config:
        config = load_config(args.config)
        datasets = config["datasets"]
        # Apply config-level defaults, then CLI overrides
        cfg_backend = config.get("backend", "ovito")
        cfg_parallel = config.get("parallel", 1)
        cfg_skip = config.get("skip_existing", False)
    else:
        # Legacy: single-dataset from CLI flags
        datasets = [
            {
                "name": Path(args.output_dir).name,
                "reference": args.reference,
                "base_dir": args.base_dir,
                "output_dir": args.output_dir,
            }
        ]
        cfg_backend = "ovito"
        cfg_parallel = 1
        cfg_skip = False

    # CLI overrides take precedence
    backend = args.backend or cfg_backend
    parallel = args.parallel if args.parallel is not None else cfg_parallel
    skip_existing = (
        args.skip_existing if args.skip_existing is not None else cfg_skip
    )

    # Validate cedar prerequisites once
    if backend == "cedar" and not args.dry_run:
        from cedar_defect_analysis import check_cedar_prerequisites

        if not check_cedar_prerequisites(args.cedar_executable):
            return 1

    # ---------------------------------------------------------------
    # Process each dataset
    # ---------------------------------------------------------------
    all_start = datetime.now()
    all_results = []
    any_failed = False

    print(f"{'=' * 80}")
    print(f"Batch defect analysis — {len(datasets)} dataset(s)")
    print(f"Backend: {backend}  |  Workers: {parallel}  "
          f"|  Skip existing: {skip_existing}")
    print(f"{'=' * 80}")

    for ds_idx, ds in enumerate(datasets, start=1):
        ds_start = datetime.now()
        name = ds["name"]
        reference = ds["reference"]
        base_dir = ds["base_dir"]
        output_dir = ds["output_dir"]

        print(f"\n{'─' * 80}")
        print(f"Dataset {ds_idx}/{len(datasets)}: {name}")
        print(f"  Reference : {reference}")
        print(f"  Input dir : {base_dir}")
        print(f"  Output dir: {output_dir}")

        # Validate reference
        if not args.dry_run and not Path(reference).exists():
            logger.error("Reference file not found: %s", reference)
            any_failed = True
            continue

        # Discover cascades
        cascades = discover_cascades(base_dir)
        if not cascades:
            logger.warning("No cascades found in %s — skipping.", base_dir)
            continue

        # Look up PKA starting positions from the reference dump
        pka_ids = [c["pka_id"] for c in cascades if c.get("pka_id") is not None]
        if pka_ids:
            pka_positions = lookup_pka_positions(reference, pka_ids)
            for c in cascades:
                c["pka_position_angstrom"] = pka_positions.get(c.get("pka_id"))
        else:
            for c in cascades:
                c["pka_position_angstrom"] = None

        # Parse LAMMPS metadata
        sim_metadata = parse_lammps_inputs(base_dir)
        logger.debug("Parsed metadata: %s", sim_metadata)

        # Build tasks
        tasks = build_tasks(cascades, ds)
        total_tasks = len(tasks)
        print(f"  Cascades  : {len(cascades)} "
              f"({total_tasks} tasks including min/unmin variants)")
        print(f"{'─' * 80}")

        if args.dry_run:
            id_fmt = _id_format(len(cascades))
            for task in tasks:
                skip = _should_skip(task, skip_existing)
                status = "[SKIP]" if skip else "[DRY RUN]"
                print(
                    f"  {status} #{task['cascade_id']:{len(str(len(cascades)))}d}"
                    f" ({task['rel_path']})  {task['state']:<12s}"
                    f"  →  {task['output_prefix']}vac.dump"
                )
            continue

        # Run analysis
        results_log = run_tasks(
            tasks,
            backend=backend,
            parallel=parallel,
            reference=reference,
            cedar_executable=args.cedar_executable,
            skip_existing=skip_existing,
        )

        # Tag results with dataset name
        for r in results_log:
            r["dataset"] = name
        all_results.extend(results_log)

        # Write metadata.json
        ds_end = datetime.now()
        meta_path = write_metadata(
            output_dir, ds, sim_metadata, cascades,
            results_log, backend, ds_start, ds_end,
        )

        ds_duration = (ds_end - ds_start).total_seconds()
        ok = sum(1 for r in results_log if r["success"] is True)
        fail = sum(1 for r in results_log if r["success"] is False)
        skip_n = sum(1 for r in results_log if r["success"] == "skipped")
        print(f"\n  {name} done in {ds_duration / 60:.1f} min — "
              f"OK: {ok}  Failed: {fail}  Skipped: {skip_n}")
        print(f"  Metadata: {meta_path}")

        if fail > 0:
            any_failed = True

    # ---------------------------------------------------------------
    # Global summary
    # ---------------------------------------------------------------
    if args.summary_csv and all_results:
        write_summary_csv(args.summary_csv, all_results)

    all_end = datetime.now()
    total_duration = (all_end - all_start).total_seconds()

    print(f"\n{'=' * 80}")
    print(f"All datasets complete in {total_duration / 60:.1f} minutes")
    if all_results:
        total_ok = sum(1 for r in all_results if r["success"] is True)
        total_fail = sum(1 for r in all_results if r["success"] is False)
        total_skip = sum(
            1 for r in all_results if r["success"] == "skipped"
        )
        print(f"  Total tasks: {len(all_results)}")
        print(f"  Success: {total_ok}  Failed: {total_fail}  "
              f"Skipped: {total_skip}")
    print(f"{'=' * 80}")

    if any_failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
