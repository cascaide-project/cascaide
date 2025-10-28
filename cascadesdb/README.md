# CascadesDB Metadata Toolkit

This repository provides a small Python toolkit that downloads the public metadata
records from [CascadesDB](https://cascadesdb.iaea.org/) and exposes a lightweight API
for downstream analysis.

The toolkit focuses on four goals:

- Fetch every JSON metadata record published in CascadesDB.
- Cache those JSON payloads locally so repeated analyses avoid unnecessary downloads.
- Provide a simple Python API for navigating records and their metadata fields.
- Expose helper methods to fetch large artefacts (interatomic potentials and data archives)
  on-demand, so heavy downloads happen only when explicitly requested.

## Getting started

The project has no third-party dependencies and works with Python 3.9 or newer.

1. (Optional) Create a virtual environment.
2. From the project root, run a first refresh to populate the cache:

   ```bash
   python -m cascadesdb.client --max-misses 30 --delay 0.05 refresh
   ```

   This contacts the public CascadesDB API and stores each `json/<N>/` payload under
   `~/.cache/cascadesdb/records/`. The parameters let you throttle requests and control
   how many consecutive missing IDs terminate the crawl.

3. Inspect the cached records or compute quick statistics:

   ```bash
   python -m cascadesdb.client summary
   python scripts/coverage_report.py
   ```

## Python API overview

```python
from cascadesdb import CascadesDBClient

client = CascadesDBClient()
client.refresh()  # downloads new records if any

for record in client.iter_records():
    print(record.summary())
    chemical_formula = record.material.get("chemical-formula")
    energy = record.data.get("PKA-energy")

    # Download the interatomic potential only if you need it
    # potential_path = record.download_potential()
```

- `client.get_record(record_id)` returns a `Record` object, loading from cache when possible.
- `record.data` contains the raw JSON dictionary for custom analysis.
- `record.download_potential()` and `record.download_archive()` pull the larger artefacts
  referenced from the metadata into `~/.cache/cascadesdb/potentials/` and `archives/`.
  These functions raise `DownloadError` if CascadesDB refuses access (some archives require
  authentication).

## Coverage reporting

The helper script `scripts/coverage_report.py` aggregates cached records into
simple coverage metrics (material counts, PKA energy bins, temperature ranges, etc.).
Use `--refresh` to pull new metadata before generating the report and `--json`
to emit machine-readable output.

```bash
python scripts/coverage_report.py --refresh --json > coverage.json
```

## Cache location

By default the toolkit writes into `~/.cache/cascadesdb/`. Set a custom location
by passing `--cache /path/to/cache` on the command line or by instantiating the
client with `CascadesDBClient(cache_dir="...")`.
