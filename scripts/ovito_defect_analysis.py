#!/usr/bin/env python3
"""
OVITO-based Wigner-Seitz defect analysis for LAMMPS cascade simulations.

Performs Wigner-Seitz cell analysis to identify point defects (vacancies and
self-interstitial atoms) in radiation damage cascade simulations. This script
is designed as an open-source replacement for the cedar ``-defect`` command.

Usage::

    # Basic usage
    python ovito_defect_analysis.py dump.reference dump.end.min

    # With output directory
    python ovito_defect_analysis.py dump.reference dump.end.min --output-dir ./defects

    # With visualization
    python ovito_defect_analysis.py dump.reference dump.end.min --visualize

    # Data only (no dump files)
    python ovito_defect_analysis.py dump.reference dump.end.min --no-dump-files

Requirements:
    - ovito (``pip install ovito``)
    - numpy
    - pandas (optional, for DataFrame output)
    - matplotlib (optional, for visualization)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _minimum_image_distance(positions, reference, box_lengths):
    """
    Compute distances under the minimum image convention.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) array of positions.
    reference : np.ndarray
        (3,) array – single reference point.
    box_lengths : np.ndarray
        (3,) array of box dimensions.

    Returns
    -------
    np.ndarray
        (N,) array of distances.
    """
    delta = positions - reference
    for dim in range(3):
        delta[:, dim] -= box_lengths[dim] * np.round(delta[:, dim] / box_lengths[dim])
    return np.linalg.norm(delta, axis=1)


def write_lammps_dump(
    filepath,
    atom_ids,
    atom_types,
    positions,
    timestep,
    box_bounds,
    extra_columns=None,
    extra_names=None,
):
    """
    Write atoms to a LAMMPS dump file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    atom_ids : np.ndarray
        (N,) array of integer atom IDs.
    atom_types : np.ndarray
        (N,) array of integer atom types.
    positions : np.ndarray
        (N, 3) array of x, y, z positions.
    timestep : int
        Simulation timestep.
    box_bounds : np.ndarray
        (3, 2) array ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]``.
    extra_columns : list of np.ndarray, optional
        Additional per-atom data columns.
    extra_names : list of str, optional
        Header names for the extra columns.
    """
    n_atoms = len(atom_ids)

    col_header = "ITEM: ATOMS id type x y z"
    if extra_names:
        col_header += " " + " ".join(extra_names)

    with open(filepath, "w") as f:
        f.write(f"ITEM: TIMESTEP\n{timestep}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for dim in range(3):
            f.write(f"{box_bounds[dim, 0]:.15g} {box_bounds[dim, 1]:.15g}\n")
        f.write(f"{col_header}\n")

        for i in range(n_atoms):
            parts = [
                f"{int(atom_ids[i])}",
                f"{int(atom_types[i])}",
                f"{positions[i, 0]:.6g}",
                f"{positions[i, 1]:.6g}",
                f"{positions[i, 2]:.6g}",
            ]
            if extra_columns:
                for col in extra_columns:
                    parts.append(f"{col[i]:.6g}")
            f.write(" ".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Core Analysis
# ---------------------------------------------------------------------------


def analyze_defects(
    reference_file: str,
    displaced_file: str,
    output_dir: str = ".",
    output_prefix: str = "",
    write_dump_files: bool = True,
    return_data: bool = False,
    as_dataframe: bool = False,
) -> dict:
    """
    Perform Wigner-Seitz defect analysis using OVITO.

    Compares a displaced atomic configuration against a reference lattice to
    identify vacancies and self-interstitial atoms (SIAs).  Produces the same
    three output files as ``cedar -defect``:

    * ``vac.dump``      – vacancy lattice-site positions
    * ``sia.dump``      – interstitial atom positions (from the displaced config)
    * ``sia_site.dump`` – lattice-site positions that gained extra atoms

    Parameters
    ----------
    reference_file : str
        Path to reference LAMMPS dump file (perfect lattice).
    displaced_file : str
        Path to displaced LAMMPS dump file (after cascade / minimization).
    output_dir : str
        Directory for output files (default: current directory).
    output_prefix : str
        Prefix for output filenames (e.g. ``"cascade_001_"``).
    write_dump_files : bool
        If True, write vac.dump, sia.dump, sia_site.dump.
    return_data : bool
        If True, include defect data arrays in the return dictionary.
    as_dataframe : bool
        If True *and* ``return_data=True``, return pandas DataFrames
        instead of NumPy arrays.

    Returns
    -------
    dict
        Always contains:
            ``n_vacancies``, ``n_sias``, ``n_sia_sites``, ``timestep``,
            ``box_bounds`` (3×2 ndarray).
        When *write_dump_files*:
            ``vac_file``, ``sia_file``, ``sia_site_file``.
        When *return_data*:
            ``vac_data``, ``sia_data``, ``sia_site_data``
            (NumPy structured arrays or pandas DataFrames).
    """
    from ovito.io import import_file as ovito_import
    from ovito.modifiers import WignerSeitzAnalysisModifier
    from ovito.pipeline import FileSource, ReferenceConfigurationModifier

    reference_file = str(Path(reference_file).resolve())
    displaced_file = str(Path(displaced_file).resolve())
    output_dir = Path(output_dir)

    if write_dump_files:
        output_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_prefix if output_prefix else ""

    logger.info("Reference : %s", reference_file)
    logger.info("Displaced : %s", displaced_file)

    # ==================================================================
    # Pipeline 1 – Reference-site mode  (output_displaced=False)
    # ------------------------------------------------------------------
    # The output contains one entry per reference lattice site with its
    # occupancy count.  Sites with occupancy 0 are vacancies; sites with
    # occupancy > 1 contain interstitial atoms.
    # ==================================================================
    logger.info("Running Wigner-Seitz analysis (reference-site mode) ...")
    pipeline_ref = ovito_import(displaced_file)
    ws_ref = WignerSeitzAnalysisModifier(output_displaced=False)
    ws_ref.affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
    ref_source1 = FileSource()
    ref_source1.load(reference_file)
    ws_ref.reference = ref_source1
    pipeline_ref.modifiers.append(ws_ref)
    data_ref = pipeline_ref.compute()

    n_vac = data_ref.attributes["WignerSeitz.vacancy_count"]
    n_int = data_ref.attributes["WignerSeitz.interstitial_count"]
    logger.info("Wigner-Seitz counts: %d vacancies, %d interstitials", n_vac, n_int)

    # Reference-site arrays
    occupancy_ref = np.asarray(data_ref.particles["Occupancy"])
    positions_ref = np.asarray(data_ref.particles["Position"])
    ids_ref = np.asarray(data_ref.particles["Particle Identifier"])
    types_ref = np.asarray(data_ref.particles["Particle Type"])

    # Simulation metadata
    timestep = int(data_ref.attributes.get("Timestep", 0))
    cell_matrix = np.asarray(data_ref.cell[...])  # 3×4: columns = cell vectors + origin
    origin = cell_matrix[:, 3]
    box_bounds = np.zeros((3, 2))
    box_lengths = np.zeros(3)
    for dim in range(3):
        box_bounds[dim, 0] = origin[dim]
        box_bounds[dim, 1] = origin[dim] + cell_matrix[dim, dim]
        box_lengths[dim] = cell_matrix[dim, dim]

    # --- Vacancy sites (occupancy == 0) ---
    vac_mask = occupancy_ref == 0
    vac_ids = ids_ref[vac_mask]
    vac_types = types_ref[vac_mask]
    vac_positions = positions_ref[vac_mask]

    # --- SIA sites (occupancy > 1) ---
    sia_site_mask = occupancy_ref > 1
    sia_site_ids = ids_ref[sia_site_mask]
    sia_site_types = types_ref[sia_site_mask]
    sia_site_positions = positions_ref[sia_site_mask]
    n_sia_sites = int(np.sum(sia_site_mask))

    # Build fast lookup: reference site ID → reference position
    ref_id_to_idx = {int(ids_ref[i]): i for i in range(len(ids_ref))}

    # ==================================================================
    # Pipeline 2 – Displaced-atom mode  (output_displaced=True)
    # ------------------------------------------------------------------
    # The output retains the displaced atoms.  Each atom is annotated with
    # the Occupancy of the reference site it was assigned to, the
    # Site Index, and (if available) the Site Identifier.
    # ==================================================================
    logger.info("Running Wigner-Seitz analysis (displaced-atom mode) ...")
    pipeline_disp = ovito_import(displaced_file)
    ws_disp = WignerSeitzAnalysisModifier(output_displaced=True)
    ws_disp.affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
    ref_source2 = FileSource()
    ref_source2.load(reference_file)
    ws_disp.reference = ref_source2
    pipeline_disp.modifiers.append(ws_disp)
    data_disp = pipeline_disp.compute()

    occ_disp = np.asarray(data_disp.particles["Occupancy"])
    ids_disp = np.asarray(data_disp.particles["Particle Identifier"])
    types_disp = np.asarray(data_disp.particles["Particle Type"])
    positions_disp = np.asarray(data_disp.particles["Position"])

    # Resolve site identifiers
    if "Site Identifier" in data_disp.particles:
        site_ids_disp = np.asarray(data_disp.particles["Site Identifier"])
    else:
        # Fallback: translate Site Index → reference Particle Identifier
        site_index = np.asarray(data_disp.particles["Site Index"])
        site_ids_disp = ids_ref[site_index]

    # Collect optional per-atom properties (e.g. c_myKE, c_myPE).
    # OVITO may lowercase property names from LAMMPS dump headers, so we
    # search case-insensitively and report the canonical (mixed-case) name.
    _available_props = set(data_disp.particles.keys())
    _avail_lower = {k.lower(): k for k in _available_props}
    extra_names: List[str] = []
    extra_arrays: List[np.ndarray] = []
    for canonical_name in ("c_myKE", "c_myPE"):
        actual = _avail_lower.get(canonical_name.lower())
        if actual is not None:
            extra_names.append(canonical_name)  # use canonical name in output
            extra_arrays.append(np.asarray(data_disp.particles[actual]))

    # ==================================================================
    # Identify SIA atoms
    # ------------------------------------------------------------------
    # For every multi-occupied reference site (occupancy N > 1) exactly
    # N − 1 atoms are interstitials.  The "native" occupant is the atom
    # whose Particle Identifier matches the Site Identifier of the
    # reference site.  If the native left (replacement collision), the
    # atom closest to the reference-site position (minimum-image
    # convention) is treated as occupant and the rest as interstitials.
    # ==================================================================
    multi_occ_site_ids = np.unique(site_ids_disp[occ_disp > 1])
    sia_atom_indices: List[int] = []

    for site_id in multi_occ_site_ids:
        at_site = np.where(site_ids_disp == site_id)[0]
        n_at_site = len(at_site)
        if n_at_site <= 1:
            continue  # should not happen

        pids = ids_disp[at_site]
        native_mask = pids == site_id

        if np.any(native_mask):
            # Native occupant present – everyone else is an interstitial
            sia_atom_indices.extend(at_site[~native_mask].tolist())
        else:
            # Native absent (replacement collision).
            # Pick the atom closest to the reference site as occupant.
            ref_idx = ref_id_to_idx.get(int(site_id))
            if ref_idx is not None:
                ref_pos = positions_ref[ref_idx]
            else:
                # Shouldn't happen, but fall back to mean position
                ref_pos = positions_disp[at_site].mean(axis=0)

            dists = _minimum_image_distance(
                positions_disp[at_site], ref_pos, box_lengths
            )
            occupant_local = np.argmin(dists)
            mask = np.ones(n_at_site, dtype=bool)
            mask[occupant_local] = False
            sia_atom_indices.extend(at_site[mask].tolist())

    sia_atom_indices = np.array(sia_atom_indices, dtype=int)
    n_sia_atoms = len(sia_atom_indices)

    # Sanity check
    if n_sia_atoms != n_int:
        logger.warning(
            "SIA atom count (%d) differs from WignerSeitz.interstitial_count (%d). "
            "This may happen with complex replacement-collision chains.",
            n_sia_atoms,
            n_int,
        )

    # Extract SIA atom data
    if n_sia_atoms > 0:
        sia_ids = ids_disp[sia_atom_indices]
        sia_types = types_disp[sia_atom_indices]
        sia_positions = positions_disp[sia_atom_indices]
        sia_extra = [arr[sia_atom_indices] for arr in extra_arrays]
    else:
        sia_ids = np.array([], dtype=int)
        sia_types = np.array([], dtype=int)
        sia_positions = np.empty((0, 3))
        sia_extra = [np.array([]) for _ in extra_arrays]

    logger.info(
        "Defect summary: %d vacancies, %d SIA atoms, %d SIA sites",
        len(vac_ids),
        n_sia_atoms,
        n_sia_sites,
    )

    # ==================================================================
    # Build result dictionary
    # ==================================================================
    result: Dict = {
        "n_vacancies": int(len(vac_ids)),
        "n_sias": int(n_sia_atoms),
        "n_sia_sites": int(n_sia_sites),
        "timestep": timestep,
        "box_bounds": box_bounds,
    }

    # --- Write LAMMPS dump files ---
    if write_dump_files:
        vac_file = str(output_dir / f"{prefix}vac.dump")
        sia_file = str(output_dir / f"{prefix}sia.dump")
        sia_site_file = str(output_dir / f"{prefix}sia_site.dump")

        logger.info("Writing %s  (%d atoms)", vac_file, len(vac_ids))
        write_lammps_dump(
            vac_file, vac_ids, vac_types, vac_positions, timestep, box_bounds
        )

        logger.info("Writing %s  (%d atoms)", sia_file, n_sia_atoms)
        write_lammps_dump(
            sia_file,
            sia_ids,
            sia_types,
            sia_positions,
            timestep,
            box_bounds,
            extra_columns=sia_extra if sia_extra else None,
            extra_names=extra_names if extra_names else None,
        )

        logger.info("Writing %s  (%d atoms)", sia_site_file, n_sia_sites)
        write_lammps_dump(
            sia_site_file,
            sia_site_ids,
            sia_site_types,
            sia_site_positions,
            timestep,
            box_bounds,
        )

        result["vac_file"] = vac_file
        result["sia_file"] = sia_file
        result["sia_site_file"] = sia_site_file

    # --- Return data arrays ---
    if return_data:
        base_cols_vac = ["id", "type", "x", "y", "z"]
        base_cols_sia = ["id", "type", "x", "y", "z"] + extra_names

        # Build NumPy arrays
        if len(vac_ids) > 0:
            vac_arr = np.column_stack([vac_ids, vac_types, vac_positions])
        else:
            vac_arr = np.empty((0, 5))

        if n_sia_atoms > 0:
            cols = [sia_ids, sia_types, sia_positions]
            cols.extend(sia_extra)
            sia_arr = np.column_stack(cols)
        else:
            sia_arr = np.empty((0, 5 + len(extra_names)))

        if n_sia_sites > 0:
            sia_site_arr = np.column_stack(
                [sia_site_ids, sia_site_types, sia_site_positions]
            )
        else:
            sia_site_arr = np.empty((0, 5))

        if as_dataframe:
            try:
                import pandas as pd

                vac_data = pd.DataFrame(vac_arr, columns=base_cols_vac)
                vac_data[["id", "type"]] = vac_data[["id", "type"]].astype(int)

                sia_data = pd.DataFrame(sia_arr, columns=base_cols_sia)
                if len(sia_data) > 0:
                    sia_data[["id", "type"]] = sia_data[["id", "type"]].astype(int)

                sia_site_data = pd.DataFrame(sia_site_arr, columns=base_cols_vac)
                sia_site_data[["id", "type"]] = sia_site_data[
                    ["id", "type"]
                ].astype(int)
            except ImportError:
                logger.warning(
                    "pandas not installed – falling back to NumPy arrays."
                )
                vac_data = vac_arr
                sia_data = sia_arr
                sia_site_data = sia_site_arr
        else:
            vac_data = vac_arr
            sia_data = sia_arr
            sia_site_data = sia_site_arr

        result["vac_data"] = vac_data
        result["sia_data"] = sia_data
        result["sia_site_data"] = sia_site_data

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_defects(
    vac_positions: np.ndarray,
    sia_positions: np.ndarray,
    box_bounds: Optional[np.ndarray] = None,
    output_file: Optional[str] = None,
    vac_color: str = "tab:blue",
    sia_color: str = "tab:red",
    vac_label: str = "Vacancies",
    sia_label: str = "SIAs",
    figsize: tuple = (10, 10),
    dpi: int = 150,
    title: str = "Cascade Defects",
    elev: float = 20.0,
    azim: float = -60.0,
    show: bool = False,
):
    """
    Create a 3-D scatter plot of vacancies and SIAs.

    Parameters
    ----------
    vac_positions : np.ndarray
        (N_vac, 3) vacancy coordinates.
    sia_positions : np.ndarray
        (N_sia, 3) SIA coordinates.
    box_bounds : np.ndarray, optional
        (3, 2) simulation box bounds for drawing wireframe box.
    output_file : str, optional
        Save figure to this path (PNG, PDF, …).
    vac_color, sia_color : str
        Marker colours.
    vac_label, sia_label : str
        Legend labels.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Figure resolution.
    title : str
        Plot title.
    elev, azim : float
        3-D view elevation and azimuth angles.
    show : bool
        Call ``plt.show()`` interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Plot vacancies
    if len(vac_positions) > 0:
        ax.scatter(
            vac_positions[:, 0],
            vac_positions[:, 1],
            vac_positions[:, 2],
            c=vac_color,
            marker="o",
            s=40,
            alpha=0.7,
            label=f"{vac_label} ({len(vac_positions)})",
            depthshade=True,
        )

    # Plot SIAs
    if len(sia_positions) > 0:
        ax.scatter(
            sia_positions[:, 0],
            sia_positions[:, 1],
            sia_positions[:, 2],
            c=sia_color,
            marker="^",
            s=50,
            alpha=0.7,
            label=f"{sia_label} ({len(sia_positions)})",
            depthshade=True,
        )

    # Draw simulation box wireframe
    if box_bounds is not None:
        _draw_box_wireframe(ax, box_bounds)

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
        logger.info("Saved 3D plot to %s", output_file)
    if show:
        plt.show()

    return fig


def plot_projections(
    vac_positions: np.ndarray,
    sia_positions: np.ndarray,
    output_file: Optional[str] = None,
    vac_color: str = "tab:blue",
    sia_color: str = "tab:red",
    figsize: tuple = (18, 5),
    dpi: int = 150,
    title: str = "Cascade Defect Projections",
    show: bool = False,
):
    """
    Create 2-D projection plots (xy, xz, yz) of vacancies and SIAs.

    Parameters
    ----------
    vac_positions : np.ndarray
        (N_vac, 3) vacancy coordinates.
    sia_positions : np.ndarray
        (N_sia, 3) SIA coordinates.
    output_file : str, optional
        Save figure to this path.
    vac_color, sia_color : str
        Marker colours.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Figure resolution.
    title : str
        Super-title.
    show : bool
        Call ``plt.show()`` interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    planes = [
        ("x (Å)", "y (Å)", 0, 1, "XY projection"),
        ("x (Å)", "z (Å)", 0, 2, "XZ projection"),
        ("y (Å)", "z (Å)", 1, 2, "YZ projection"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=14)

    for ax, (xlabel, ylabel, ci, cj, subtitle) in zip(axes, planes):
        if len(vac_positions) > 0:
            ax.scatter(
                vac_positions[:, ci],
                vac_positions[:, cj],
                c=vac_color,
                marker="o",
                s=30,
                alpha=0.7,
                label=f"Vac ({len(vac_positions)})",
            )
        if len(sia_positions) > 0:
            ax.scatter(
                sia_positions[:, ci],
                sia_positions[:, cj],
                c=sia_color,
                marker="^",
                s=40,
                alpha=0.7,
                label=f"SIA ({len(sia_positions)})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
        logger.info("Saved projection plot to %s", output_file)
    if show:
        plt.show()

    return fig


def _draw_box_wireframe(ax, box_bounds):
    """Draw a wireframe cuboid on *ax* from ``box_bounds`` (3×2)."""
    lo = box_bounds[:, 0]
    hi = box_bounds[:, 1]
    # 12 edges of a cuboid
    for s, e in [
        ([lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]]),
        ([lo[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]]),
        ([lo[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]]),
        ([hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]),
        ([hi[0], hi[1], hi[2]], [hi[0], lo[1], hi[2]]),
        ([hi[0], hi[1], hi[2]], [hi[0], hi[1], lo[2]]),
        ([hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]]),
        ([hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]]),
        ([lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]]),
        ([lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]]),
        ([lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]]),
        ([lo[0], lo[1], hi[2]], [lo[0], hi[1], hi[2]]),
    ]:
        ax.plot3D(*zip(s, e), color="gray", linewidth=0.5, alpha=0.4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Wigner-Seitz defect analysis on LAMMPS dump files using OVITO.  "
            "Identifies vacancies and self-interstitial atoms produced by "
            "radiation damage cascades."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s dump.reference dump.end.min\n"
            "  %(prog)s dump.reference dump.end.min --output-dir ./defects\n"
            "  %(prog)s dump.reference dump.end.min --visualize\n"
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
        "--prefix",
        default="",
        help="Prefix for output filenames.",
    )
    parser.add_argument(
        "--no-dump-files",
        action="store_true",
        help="Skip writing vac.dump / sia.dump / sia_site.dump.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate defect visualisation plots.",
    )
    parser.add_argument(
        "--plot-file",
        default=None,
        help="File path for the 3-D scatter plot (e.g. defects.png).",
    )
    parser.add_argument(
        "--projection-file",
        default=None,
        help="File path for the 2-D projection plot.",
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
    write_dumps = not args.no_dump_files
    need_data = args.visualize or args.plot_file or args.projection_file

    result = analyze_defects(
        reference_file=str(ref_path),
        displaced_file=str(disp_path),
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        write_dump_files=write_dumps,
        return_data=need_data,
        as_dataframe=False,
    )

    # Summary
    print(f"nca= 0")
    print(f"nda= {result['n_vacancies'] + result['n_sias']}")
    print(f"nvac= {result['n_vacancies']}")
    print(f"nsia= {result['n_sias']}")

    # Visualization
    if need_data:
        vac_pos = result["vac_data"][:, 2:5] if len(result["vac_data"]) > 0 else np.empty((0, 3))
        sia_pos = result["sia_data"][:, 2:5] if len(result["sia_data"]) > 0 else np.empty((0, 3))

        if args.visualize or args.plot_file:
            plot_path = args.plot_file or str(
                Path(args.output_dir) / f"{args.prefix}defects_3d.png"
            )
            visualize_defects(
                vac_pos,
                sia_pos,
                box_bounds=result["box_bounds"],
                output_file=plot_path,
            )

        if args.visualize or args.projection_file:
            proj_path = args.projection_file or str(
                Path(args.output_dir) / f"{args.prefix}defects_proj.png"
            )
            plot_projections(
                vac_pos,
                sia_pos,
                output_file=proj_path,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
