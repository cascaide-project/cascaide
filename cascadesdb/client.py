"""Client and local cache for CascadesDB metadata."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from urllib import error, request


DEFAULT_BASE_URL = "https://cascadesdb.iaea.org"
DEFAULT_USER_AGENT = (
    "CascadesDBClient/0.1 (+https://github.com/iaea/cascadesdb)"
)


def _default_cache_dir() -> Path:
    """Choose a sensible default cache directory for the current platform."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "cascadesdb"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home()
        return Path(base).expanduser() / "CascadesDB" / "Cache"
    return Path.home() / ".cache" / "cascadesdb"


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class CascadesDBError(Exception):
    """Base exception for all CascadesDB client errors."""


class RecordNotFound(CascadesDBError):
    """Raised when a requested record is not available."""


class DownloadError(CascadesDBError):
    """Raised when a remote resource cannot be downloaded."""


class _LinkCollector(HTMLParser):
    """Utility parser returning all links in an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        data = dict(attrs)
        href = data.get("href")
        if href:
            self.links.append(href)


class CascadesDBClient:
    """High-level interface for downloading CascadesDB metadata."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Optional[Path | str] = None,
        *,
        request_delay: float = 0.0,
        max_consecutive_misses: int = 30,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_delay = max(0.0, request_delay)
        self.max_consecutive_misses = max_consecutive_misses
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.records_dir = self.cache_dir / "records"
        self.potentials_dir = self.cache_dir / "potentials"
        self.archives_dir = self.cache_dir / "archives"
        _ensure_directory(self.records_dir)
        _ensure_directory(self.potentials_dir)
        _ensure_directory(self.archives_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------ Manifest
    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "available_ids": [],
                "highest_available": 0,
                "highest_checked": 0,
                "cache_version": 1,
                "last_refresh": None,
            }
        try:
            raw = json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise CascadesDBError(
                f"Failed to read manifest {self.manifest_path}: {exc}"
            ) from exc
        # ensure required keys
        raw.setdefault("available_ids", [])
        raw.setdefault("highest_available", max(raw["available_ids"], default=0))
        raw.setdefault("highest_checked", raw.get("highest_available", 0))
        raw.setdefault("cache_version", 1)
        raw.setdefault("last_refresh", None)
        return raw

    def _save_manifest(self) -> None:
        data = dict(self._manifest)
        data["available_ids"] = sorted(set(data.get("available_ids", [])))
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="manifest", suffix=".json", dir=self.cache_dir)
        try:
            with os.fdopen(tmp_fd, "w") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, self.manifest_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------ Public API
    def refresh(
        self,
        *,
        start_id: Optional[int] = None,
        force: bool = False,
        until_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Download new records and update the local cache.

        Returns a dictionary with counts of downloaded, skipped, and missing records.
        """
        if force:
            start_id = start_id or 1
            known_ids: Set[int] = set()
        else:
            known_ids = set(int(i) for i in self._manifest.get("available_ids", []))
            start_id = start_id or (
                max(known_ids) + 1 if known_ids else 1
            )

        downloaded = 0
        skipped = 0
        missing = 0
        consecutive_misses = 0
        record_id = max(1, start_id)

        while consecutive_misses < self.max_consecutive_misses:
            if until_id is not None and record_id > until_id:
                break

            if not force and record_id in known_ids and self._record_cache_path(record_id).exists():
                skipped += 1
                consecutive_misses = 0
                record_id += 1
                continue

            try:
                data = self._download_record(record_id)
            except RecordNotFound:
                missing += 1
                consecutive_misses += 1
            else:
                self._write_record_cache(record_id, data)
                known_ids.add(record_id)
                downloaded += 1
                consecutive_misses = 0
            record_id += 1
            if self.request_delay:
                time.sleep(self.request_delay)

        if known_ids:
            highest_available = max(known_ids)
        else:
            highest_available = 0

        self._manifest["available_ids"] = sorted(known_ids)
        self._manifest["highest_available"] = highest_available
        self._manifest["highest_checked"] = max(
            record_id - 1, self._manifest.get("highest_checked", 0)
        )
        self._manifest["last_refresh"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest()

        return {
            "downloaded": downloaded,
            "skipped": skipped,
            "missing": missing,
            "last_checked_id": record_id - 1,
            "highest_available": highest_available,
        }

    def list_record_ids(self) -> List[int]:
        """Return sorted list of record IDs available in the cache."""
        return sorted(int(i) for i in self._manifest.get("available_ids", []))

    def iter_records(self) -> Iterator["Record"]:
        """Iterate through cached records as Record objects."""
        for record_id in self.list_record_ids():
            yield self.get_record(record_id, fetch_if_missing=False)

    def get_record(
        self,
        record_id: int,
        *,
        fetch_if_missing: bool = True,
        force: bool = False,
    ) -> "Record":
        """Return a record, downloading it if necessary."""
        cache_path = self._record_cache_path(record_id)
        if not force and cache_path.exists():
            with cache_path.open("r") as handle:
                data = json.load(handle)
            return Record(record_id, data, self)

        if not fetch_if_missing:
            raise RecordNotFound(f"Record {record_id} is not cached locally.")

        data = self._download_record(record_id)
        self._write_record_cache(record_id, data)
        ids = set(int(i) for i in self._manifest.get("available_ids", []))
        ids.add(record_id)
        self._manifest["available_ids"] = sorted(ids)
        self._manifest["highest_available"] = max(
            ids, default=self._manifest.get("highest_available", 0)
        )
        self._save_manifest()
        return Record(record_id, data, self)

    # ------------------------------------------------------------------ Downloads
    def download_potential(
        self,
        record: "Record",
        *,
        force: bool = False,
    ) -> Path:
        """Download the interatomic potential file for a record."""
        potential_info = record.data.get("potential")
        if not potential_info:
            raise DownloadError(f"Record {record.record_id} does not include potential metadata.")

        filename = potential_info.get("filename")
        potential_uri = potential_info.get("uri")
        if not filename or not potential_uri:
            raise DownloadError(f"Record {record.record_id} has incomplete potential metadata.")

        target_path = self.potentials_dir / filename
        if target_path.exists() and not force:
            return target_path

        potential_url = self._resolve_potential_url(potential_uri, filename)
        content = self._download_binary(potential_url)
        with target_path.open("wb") as handle:
            handle.write(content)
        return target_path

    def download_archive(
        self,
        record: "Record",
        *,
        force: bool = False,
    ) -> Path:
        """Download the archive referenced by a record."""
        archive_name = record.data.get("archive-name") or record.data.get("data", {}).get("archive_name")
        if not archive_name:
            raise DownloadError(f"Record {record.record_id} does not specify an archive filename.")

        archive_path = self.archives_dir / archive_name
        if archive_path.exists() and not force:
            return archive_path

        archive_url = self._resolve_archive_url(archive_name)
        content = self._download_binary(archive_url)
        with archive_path.open("wb") as handle:
            handle.write(content)
        return archive_path

    # ------------------------------------------------------------------ Internals
    def _record_cache_path(self, record_id: int) -> Path:
        return self.records_dir / f"{record_id}.json"

    def _download_record(self, record_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/cdbmeta/cdbrecord/json/{record_id}/"
        req = request.Request(url, headers=self._headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except error.HTTPError as exc:
            if exc.code == 404:
                raise RecordNotFound(f"Record {record_id} not found.") from exc
            raise DownloadError(f"Failed to download record {record_id}: {exc}") from exc
        except error.URLError as exc:
            raise DownloadError(f"Network error downloading record {record_id}: {exc}") from exc

        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DownloadError(f"Record {record_id} returned invalid JSON.") from exc

    def _write_record_cache(self, record_id: int, data: Dict[str, Any]) -> None:
        cache_path = self._record_cache_path(record_id)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"record-{record_id}-", suffix=".json", dir=self.records_dir)
        try:
            with os.fdopen(tmp_fd, "w") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, cache_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _resolve_potential_url(self, potential_uri: str, filename: str) -> str:
        if potential_uri.endswith("/"):
            potential_uri = potential_uri[:-1]
        if potential_uri.startswith("/"):
            potential_uri = f"{self.base_url}{potential_uri}"

        # Fetch the HTML page and look for a direct data link.
        req = request.Request(potential_uri + "/", headers={"User-Agent": self._headers["User-Agent"]})
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except error.HTTPError as exc:
            raise DownloadError(f"Failed to resolve potential URL {potential_uri}: {exc}") from exc
        except error.URLError as exc:
            raise DownloadError(f"Network error while resolving potential URL {potential_uri}: {exc}") from exc

        parser = _LinkCollector()
        parser.feed(html)
        for href in parser.links:
            if href.endswith(filename):
                return href if href.startswith("http") else f"{self.base_url}{href}"

        # Fallback guess
        guess = f"{self.base_url}/data/cdb-pot/{filename}"
        return guess

    def _resolve_archive_url(self, archive_name: str) -> str:
        guessed_paths = [
            f"{self.base_url}/data/cdb-md/{archive_name}",
            f"{self.base_url}/data/cdb-data/{archive_name}",
            f"{self.base_url}/data/cdb-archive/{archive_name}",
        ]
        for url in guessed_paths:
            req = request.Request(url, method="HEAD", headers={"User-Agent": self._headers["User-Agent"]})
            try:
                with request.urlopen(req, timeout=self.timeout):
                    return url
            except error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise DownloadError(
                        f"Archive {archive_name} exists but access is forbidden at {url}."
                    ) from exc
                if exc.code != 404:
                    continue
            except error.URLError:
                continue
        # If all guesses fail, fall back to the most likely and let caller see the error.
        return guessed_paths[0]

    def _download_binary(self, url: str) -> bytes:
        headers = {"User-Agent": self._headers["User-Agent"]}
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except error.HTTPError as exc:
            raise DownloadError(f"Failed to download binary resource {url}: {exc}") from exc
        except error.URLError as exc:
            raise DownloadError(f"Network error while downloading {url}: {exc}") from exc


@dataclass
class Record(Mapping[str, Any]):
    """Representation of a CascadesDB metadata record."""

    record_id: int
    data: Dict[str, Any]
    client: CascadesDBClient

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def qid(self) -> Optional[str]:
        return self.data.get("qid")

    @property
    def material(self) -> Optional[Dict[str, Any]]:
        return self.data.get("material")

    @property
    def potential(self) -> Optional[Dict[str, Any]]:
        return self.data.get("potential")

    @property
    def archive_name(self) -> Optional[str]:
        return self.data.get("archive-name") or self.data.get("data", {}).get("archive_name")

    def download_potential(self, *, force: bool = False) -> Path:
        return self.client.download_potential(self, force=force)

    def download_archive(self, *, force: bool = False) -> Path:
        return self.client.download_archive(self, force=force)

    def summary(self) -> str:
        material = self.material or {}
        chem = material.get("chemical-formula") or material.get("formula") or "?"
        structure = material.get("structure") or ""
        energy = self.data.get("PKA-energy") or self.data.get("PKA", {}).get("energy")
        temperature = self.data.get("initial-temperature") or self.data.get("initial_temperature")
        code = self.data.get("code", {}).get("name")
        return (
            f"Record {self.record_id} ({self.qid or 'unknown'}): {chem} {structure} | "
            f"PKA={energy} keV | T0={temperature} K | Code={code or 'n/a'}"
        )


def _cli_summary(client: CascadesDBClient) -> None:
    records = list(client.iter_records())
    if not records:
        print("No records cached. Run `python -m cascadesdb.client refresh` first.")
        return

    print(f"Cached records: {len(records)}")
    by_material: Dict[str, int] = {}
    for record in records:
        material = record.material or {}
        key = material.get("chemical-formula") or material.get("formula") or "unknown"
        by_material[key] = by_material.get(key, 0) + 1
    print("Records by material:")
    for material, count in sorted(by_material.items(), key=lambda item: item[1], reverse=True):
        print(f"  {material}: {count}")


def _cli_refresh(client: CascadesDBClient) -> None:
    stats = client.refresh()
    print(
        "refresh complete -> "
        f"downloaded={stats['downloaded']} | skipped={stats['skipped']} | "
        f"missing={stats['missing']} | highest_available={stats['highest_available']}"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interact with CascadesDB metadata.")
    parser.add_argument(
        "--cache",
        dest="cache_dir",
        help="Directory for cached JSON and files (default: platform cache).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between network requests (seconds).",
    )
    parser.add_argument(
        "--max-misses",
        type=int,
        default=30,
        help="Stop refreshing after this many consecutive missing records.",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("refresh", help="Download new metadata records.")
    subparsers.add_parser("summary", help="Print basic cache summary.")
    show_parser = subparsers.add_parser("show", help="Display one record.")
    show_parser.add_argument("record_id", type=int, help="Numeric record identifier.")

    args = parser.parse_args(argv)
    client = CascadesDBClient(
        cache_dir=args.cache_dir,
        request_delay=args.delay,
        max_consecutive_misses=args.max_misses,
    )

    command = args.command or "summary"
    if command == "refresh":
        _cli_refresh(client)
    elif command == "summary":
        _cli_summary(client)
    elif command == "show":
        record = client.get_record(args.record_id)
        print(record.summary())
        print(json.dumps(record.data, indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
