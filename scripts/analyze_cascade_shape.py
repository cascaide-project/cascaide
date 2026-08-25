#!/usr/bin/env python3
"""Summarize cascade-defect extents relative to the applied PKA direction."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

CASCADE_FIELDS = [
    "dataset_name", "cascade_id", "source_path", "state", "status", "reason",
    "pka_energy_eV", "pka_energy_keV", "n_vacancies", "n_sias", "n_defects",
    "forward_extent_angstrom", "backward_extent_angstrom",
    "transverse_extent_angstrom", "asymmetry_A", "elongation_E",
]
SUMMARY_FIELDS = [
    "dataset_name", "n_cascades", "n_valid", "n_skipped",
    "median_energy_keV", "median_asymmetry_A", "median_elongation_E",
    "p95_forward_extent_angstrom", "p95_backward_extent_angstrom",
    "p95_transverse_extent_angstrom",
]


def read_lammps_positions(path):
    """Read x/y/z positions from a single-frame LAMMPS dump."""
    positions, indices = [], None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ITEM: ATOMS"):
                if indices is not None:
                    raise ValueError("multiple frames are not supported")
                columns = line.split()[2:]
                try:
                    indices = tuple(columns.index(axis) for axis in ("x", "y", "z"))
                except ValueError as exc:
                    raise ValueError("ATOMS header must contain x, y, and z") from exc
            elif line.startswith("ITEM:") or indices is None:
                continue
            elif line.strip():
                values = line.split()
                try:
                    positions.append(tuple(float(values[i]) for i in indices))
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"invalid atom row: {line.rstrip()}") from exc
    if indices is None:
        raise ValueError("missing ITEM: ATOMS header")
    return positions


def calculate_extents(positions, pka_position, pka_direction):
    """Calculate forward F, backward B, transverse T, asymmetry A, and elongation E."""
    norm = math.sqrt(sum(float(value) ** 2 for value in pka_direction))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("PKA direction must be a finite, nonzero vector")
    direction = tuple(float(value) / norm for value in pka_direction)
    projections, transverse = [], []
    for position in positions:
        displacement = tuple(float(position[i]) - float(pka_position[i]) for i in range(3))
        projection = sum(displacement[i] * direction[i] for i in range(3))
        projections.append(projection)
        transverse.append(math.sqrt(max(
            0.0, sum(value * value for value in displacement) - projection * projection
        )))
    if not projections:
        raise ValueError("defect cloud is empty")
    forward = max(0.0, max(projections))
    backward = max(0.0, -min(projections))
    transverse_extent = max(transverse)
    longitudinal = forward + backward
    return {
        "forward_extent_angstrom": forward,
        "backward_extent_angstrom": backward,
        "transverse_extent_angstrom": transverse_extent,
        "asymmetry_A": (forward - backward) / longitudinal if longitudinal else math.nan,
        "elongation_E": longitudinal / (2 * transverse_extent) if transverse_extent else math.inf,
    }


def _xyz(mapping, name):
    try:
        values = tuple(float(mapping[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unavailable or invalid") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} is unavailable or invalid")
    return values


def _output_path(dataset_dir, state_data, kind):
    keys = {"vacancy": ("vacancy_dump", "vacancies", "vac"),
            "sia": ("sia_dump", "sias", "sia")}[kind]
    outputs = state_data.get("output_files", {})
    if isinstance(outputs, list):
        suffix = "_vac.dump" if kind == "vacancy" else "_sia.dump"
        matches = [value for value in outputs if str(value).endswith(suffix)]
        if len(matches) == 1:
            path = Path(matches[0])
            return path if path.is_absolute() else dataset_dir / path
        raise ValueError(f"expected one {kind} dump in output_files")
    for key in keys:
        if outputs.get(key):
            path = Path(outputs[key])
            return path if path.is_absolute() else dataset_dir / path
    raise ValueError(f"{kind} dump is not recorded in output_files")


def analyze_cascade(dataset_dir, dataset_name, cascade, state):
    row = {field: "" for field in CASCADE_FIELDS}
    row.update(dataset_name=dataset_name, cascade_id=cascade.get("cascade_id", ""),
               source_path=cascade.get("source_path", ""), state=state, status="skipped")
    try:
        energy = float(cascade.get("pka_energy_eV"))
        if not math.isfinite(energy):
            raise ValueError
        row.update(pka_energy_eV=energy, pka_energy_keV=energy / 1000)
    except (TypeError, ValueError):
        row["reason"] = "PKA energy is unavailable or invalid"
        return row
    try:
        position = _xyz(cascade.get("pka_position_angstrom", {}), "PKA position")
        direction = _xyz(cascade.get("pka_direction_vector", {}), "PKA direction")
        state_data = cascade.get("states", {}).get(state, {})
        if state_data.get("success") is not True:
            raise ValueError(f"{state} state is not marked successful")
        vacancies = read_lammps_positions(_output_path(dataset_dir, state_data, "vacancy"))
        sias = read_lammps_positions(_output_path(dataset_dir, state_data, "sia"))
        row.update(calculate_extents(vacancies + sias, position, direction))
        row.update(status="valid", reason="", n_vacancies=len(vacancies),
                   n_sias=len(sias), n_defects=len(vacancies) + len(sias))
    except (OSError, TypeError, ValueError) as exc:
        row["reason"] = str(exc)
    return row


def discover_dataset_dirs(inputs):
    found = {}
    for item in inputs:
        item = Path(item).resolve()
        if item.is_file() and item.name == "metadata.json":
            found[item.parent] = None
        elif (item / "metadata.json").is_file():
            found[item] = None
        elif item.is_dir():
            for metadata in sorted(item.glob("*/metadata.json")):
                found[metadata.parent] = None
        else:
            raise ValueError(f"not a metadata file or dataset path: {item}")
    if not found:
        raise ValueError("no metadata.json files found")
    return list(found)


def analyze_datasets(dataset_dirs, state):
    rows = []
    for dataset_dir in dataset_dirs:
        with (dataset_dir / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        name = metadata.get("dataset_name") or dataset_dir.name
        rows.extend(analyze_cascade(dataset_dir, name, cascade, state)
                    for cascade in metadata.get("cascades", []))
    return rows


def percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return (ordered[lower] if lower == upper else
            ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def summarize_by_dataset(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset_name"]].append(row)
    summaries = []
    for name, group in grouped.items():
        valid = [row for row in group if row["status"] == "valid"]
        finite_e = [row for row in valid if math.isfinite(float(row["elongation_E"]))]
        summaries.append({
            "dataset_name": name, "n_cascades": len(group), "n_valid": len(valid),
            "n_skipped": len(group) - len(valid),
            "median_energy_keV": statistics.median(row["pka_energy_keV"] for row in valid) if valid else math.nan,
            "median_asymmetry_A": statistics.median(row["asymmetry_A"] for row in valid) if valid else math.nan,
            "median_elongation_E": statistics.median(row["elongation_E"] for row in finite_e) if finite_e else math.nan,
            "p95_forward_extent_angstrom": percentile([row["forward_extent_angstrom"] for row in valid], .95),
            "p95_backward_extent_angstrom": percentile([row["backward_extent_angstrom"] for row in valid], .95),
            "p95_transverse_extent_angstrom": percentile([row["transverse_extent_angstrom"] for row in valid], .95),
        })
    return sorted(summaries, key=lambda row: (
        not math.isfinite(row["median_energy_keV"]),
        row["median_energy_keV"] if math.isfinite(row["median_energy_keV"]) else 0,
        row["dataset_name"],
    ))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summaries):
    headings = ["Dataset", "N", "Valid", "Skipped", "Median energy (keV)",
                "Median A", "Median E", "p95 F (Ang.)", "p95 B (Ang.)", "p95 T (Ang.)"]
    lines = ["# Cascade shape summary", "",
             "Each cascade has equal weight; defect atoms are not pooled across cascades.", "",
             "| " + " | ".join(headings) + " |",
             "| " + " | ".join(["---"] * len(headings)) + " |"]
    for row in summaries:
        values = [row[key] for key in SUMMARY_FIELDS]
        lines.append("| " + " | ".join(
            f"{value:.4g}" if isinstance(value, float) and math.isfinite(value)
            else "NA" if isinstance(value, float) else str(value) for value in values
        ) + " |")
    return "\n".join(lines) + "\n"


def create_plots(rows, output_dir, dpi, extension):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required unless --no-plots is used") from exc
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "valid" and math.isfinite(float(row["elongation_E"])):
            grouped[row["dataset_name"]].append(row)
    groups = sorted(grouped.items(), key=lambda item: statistics.median(
        row["pka_energy_keV"] for row in item[1]))

    def scatter(filename, xkey, ykey, xlabel, ylabel, reference):
        figure, axis = plt.subplots(figsize=(7.2, 5.2))
        for name, group in groups:
            axis.scatter([row[xkey] for row in group], [row[ykey] for row in group],
                         s=15, alpha=.55, label=name)
        axis.axhline(reference, color="0.35", linestyle="--", linewidth=1)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=.2)
        if groups:
            axis.legend(title="Dataset", fontsize="small", frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / f"{filename}.{extension}", dpi=dpi)
        plt.close(figure)

    scatter("asymmetry_vs_energy", "pka_energy_keV", "asymmetry_A",
            "PKA energy (keV)", "Asymmetry A", 0)
    scatter("elongation_vs_energy", "pka_energy_keV", "elongation_E",
            "PKA energy (keV)", "Elongation E", 1)
    scatter("asymmetry_vs_elongation", "elongation_E", "asymmetry_A",
            "Elongation E", "Asymmetry A", 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--state", choices=("minimized", "unminimized"), default="minimized")
    parser.add_argument("--plot-format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="return nonzero if any cascade is skipped")
    args = parser.parse_args(argv)
    try:
        dataset_dirs = discover_dataset_dirs(args.inputs)
        rows = analyze_datasets(dataset_dirs, args.state)
        summaries = summarize_by_dataset(rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "cascade_shape_metrics.csv", rows, CASCADE_FIELDS)
        write_csv(args.output_dir / "energy_band_summary.csv", summaries, SUMMARY_FIELDS)
        (args.output_dir / "energy_band_summary.md").write_text(
            render_markdown(summaries), encoding="utf-8")
        if not args.no_plots:
            create_plots(rows, args.output_dir, args.dpi, args.plot_format)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    skipped = sum(row["status"] != "valid" for row in rows)
    print(f"Analyzed {len(rows)} cascades from {len(dataset_dirs)} datasets: "
          f"{len(rows) - skipped} valid, {skipped} skipped")
    print(f"Wrote results to {args.output_dir}")
    return 1 if args.strict and skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
