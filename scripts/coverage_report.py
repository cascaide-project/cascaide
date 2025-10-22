#!/usr/bin/env python3
"""Generate quick coverage statistics for CascadesDB metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List, Optional

import sys
from pathlib import Path

SYS_ROOT = Path(__file__).resolve().parent.parent
if str(SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_ROOT))

from cascadesdb import CascadesDBClient, Record


def _collect_records(client: CascadesDBClient, refresh: bool, force: bool) -> List[Record]:
    if refresh:
        stats = client.refresh(force=force)
        print(
            "Refresh stats -> "
            f"downloaded={stats['downloaded']} | skipped={stats['skipped']} | "
            f"missing={stats['missing']} | highest_available={stats['highest_available']}"
        )
    return list(client.iter_records())


def _bin_index(value: Optional[float], edges: List[float]) -> Optional[int]:
    if value is None:
        return None
    for idx, edge in enumerate(edges):
        if value < edge:
            return idx
    return len(edges)


def _bucket_label(index: int, edges: List[float]) -> str:
    if index < len(edges):
        lower = edges[index - 1] if index else 0.0
        upper = edges[index]
        return f"[{lower:g}, {upper:g})"
    lower = edges[-1] if edges else 0.0
    return f"[{lower:g}, +inf)"


def build_statistics(records: Iterable[Record]) -> Dict[str, object]:
    records = list(records)
    materials = Counter()
    codes = Counter()
    energy_values: List[float] = []
    temp_values: List[float] = []
    energy_bins = [1, 5, 10, 20, 50, 100, 200, 500]
    energy_hist: Counter[int] = Counter()
    temp_bins = [10, 50, 100, 300, 600, 1000]
    temp_hist: Counter[int] = Counter()

    for record in records:
        material = record.material or {}
        formula = material.get("chemical-formula") or material.get("formula")
        if formula:
            materials[formula] += 1

        code_name = record.data.get("code", {}).get("name")
        if code_name:
            codes[code_name] += 1

        energy = record.data.get("PKA-energy")
        if energy is None:
            try:
                energy = record.data.get("PKA", {}).get("energy")
            except AttributeError:
                energy = None
        if isinstance(energy, (int, float)):
            energy_values.append(float(energy))
            index = _bin_index(float(energy), energy_bins)
            if index is not None:
                energy_hist[index] += 1

        temperature = record.data.get("initial-temperature")
        if temperature is None:
            try:
                temperature = record.data.get("initial_temperature")
            except AttributeError:
                temperature = None
        if isinstance(temperature, (int, float)):
            temp_values.append(float(temperature))
            index = _bin_index(float(temperature), temp_bins)
            if index is not None:
                temp_hist[index] += 1

    summary: Dict[str, object] = {
        "total_records": len(records),
        "materials": materials.most_common(),
        "codes": codes.most_common(),
        "pka_energy": {
            "count": len(energy_values),
            "min": min(energy_values) if energy_values else None,
            "max": max(energy_values) if energy_values else None,
            "average": mean(energy_values) if energy_values else None,
            "histogram": {
                _bucket_label(index, energy_bins): energy_hist[index]
                for index in sorted(energy_hist)
            },
        },
        "initial_temperature": {
            "count": len(temp_values),
            "min": min(temp_values) if temp_values else None,
            "max": max(temp_values) if temp_values else None,
            "average": mean(temp_values) if temp_values else None,
            "histogram": {
                _bucket_label(index, temp_bins): temp_hist[index]
                for index in sorted(temp_hist)
            },
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CascadesDB metadata and produce coverage statistics."
    )
    parser.add_argument(
        "--cache",
        help="Directory for cache (defaults to user cache).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh metadata before computing statistics.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download all records even if cached.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--material",
        action="append",
        dest="materials",
        help="Only include records matching the given chemical formula. Repeat for multiple.",
    )
    parser.add_argument(
        "--code",
        action="append",
        dest="codes",
        help="Only include records matching the given simulation code name. Repeat for multiple.",
    )
    args = parser.parse_args()

    client = CascadesDBClient(cache_dir=args.cache)
    records = _collect_records(client, args.refresh or args.force_refresh, args.force_refresh)

    material_args = [value.strip() for value in (args.materials or []) if value.strip()]
    code_args = [value.strip() for value in (args.codes or []) if value.strip()]
    materials_filter = {value.lower() for value in material_args} or None
    codes_filter = {value.lower() for value in code_args} or None

    if materials_filter or codes_filter:
        filtered_records = []
        for record in records:
            material = record.material or {}
            formula = (material.get("chemical-formula") or material.get("formula") or "").lower()
            code_name = (record.data.get("code", {}).get("name") or "").lower()

            if materials_filter and formula not in materials_filter:
                continue
            if codes_filter and code_name not in codes_filter:
                continue
            filtered_records.append(record)
        records = filtered_records

    stats = build_statistics(records)
    filters = {}
    if material_args:
        filters["materials"] = list(dict.fromkeys(material_args))
    if code_args:
        filters["codes"] = list(dict.fromkeys(code_args))

    if args.json_output:
        if filters:
            stats = dict(stats)
            stats["filters"] = filters
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    if material_args:
        print(f"Filtered materials: {', '.join(filters['materials'])}")
    if code_args:
        print(f"Filtered codes: {', '.join(filters['codes'])}")

    if not records:
        print("No records matched the selected filters.")
        return

    print(f"Total records: {stats['total_records']}")

    materials = stats["materials"]
    if materials:
        print("\nRecords per material:")
        for name, count in materials:
            print(f"  {name}: {count}")

    codes = stats["codes"]
    if codes:
        print("\nRecords per simulation code:")
        for name, count in codes:
            print(f"  {name}: {count}")

    energy = stats["pka_energy"]
    if energy["count"]:
        print(
            f"\nPKA energy (keV): count={energy['count']} "
            f"min={energy['min']} max={energy['max']} avg={energy['average']:.2f}"
        )
        for bucket, count in energy["histogram"].items():
            print(f"  {bucket}: {count}")

    temp = stats["initial_temperature"]
    if temp["count"]:
        print(
            f"\nInitial temperature (K): count={temp['count']} "
            f"min={temp['min']} max={temp['max']} avg={temp['average']:.2f}"
        )
        for bucket, count in temp["histogram"].items():
            print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
