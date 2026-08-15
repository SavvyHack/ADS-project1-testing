"""Download raw project data into ``data/landing``.

Pulls three sources in a single pass:

1. NYC TLC Yellow Taxi trip records (monthly PARQUET).
2. TLC taxi zone lookup table and zone shapefile.
3. BTS Reporting Carrier On-Time Performance (monthly ZIP), filtered down to
   flights touching the NYC airports before being written to disk.

The BTS archives expand to roughly 250 MB of CSV per month, the vast majority
of which is irrelevant to this project. Each month is therefore filtered to
NYC-airport flights and a fixed column subset, written as PARQUET, and the
extracted CSV discarded. This keeps the landing area to a few hundred MB
rather than several GB.

All downloads are idempotent: existing outputs are skipped unless
``--overwrite`` is passed, so an interrupted run can simply be repeated.

Usage
-----
    python scripts/download.py --start 2023-01 --end 2024-06
    python scripts/download.py --start 2023-01 --end 2024-06 --skip-bts
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TLC_MISC_URL = "https://d37ci6vzurychx.cloudfront.net/misc"
BTS_BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
)

#: Taxi colour to download. Yellow is the only colour with meaningful airport
#: volume, and the only one permitted to street-hail city-wide.
TAXI_COLOUR = "yellow"

#: BTS airport codes retained. EWR is kept for context even though it sits
#: outside the TLC taxi zones, since it competes for the same passengers.
NYC_AIRPORTS = ("JFK", "LGA", "EWR")

#: BTS columns retained. Everything else in the raw feed is dropped at ingest.
BTS_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelayMinutes",
    "CRSArrTime",
    "ArrTime",
    "ArrDelayMinutes",
    "Cancelled",
    "Diverted",
    "Distance",
]

CHUNK_SIZE = 1 << 20  # 1 MiB
REQUEST_TIMEOUT = (10, 300)  # (connect, read) seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def build_session() -> requests.Session:
    """Return a ``requests`` session with retry/backoff on transient failures.

    Both hosts intermittently return 5xx responses or drop long connections,
    so retries are essential for an unattended multi-month download.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    # transtats rejects the default urllib user agent.
    session.headers.update({"User-Agent": "Mozilla/5.0 (MAST30034 project)"})
    return session


def month_range(start: str, end: str) -> Iterator[tuple[int, int]]:
    """Yield ``(year, month)`` pairs inclusive of both endpoints.

    Args:
        start: First month as ``"YYYY-MM"``.
        end: Last month as ``"YYYY-MM"``.

    Yields:
        Successive ``(year, month)`` tuples.

    Raises:
        ValueError: If ``end`` precedes ``start`` or either fails to parse.
    """
    try:
        start_year, start_month = (int(part) for part in start.split("-"))
        end_year, end_month = (int(part) for part in end.split("-"))
    except ValueError as exc:
        raise ValueError("Months must be formatted as YYYY-MM.") from exc

    if (end_year, end_month) < (start_year, start_month):
        raise ValueError(f"End month {end} precedes start month {start}.")

    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    overwrite: bool = False,
) -> bool:
    """Stream ``url`` to ``destination``.

    Downloads to a temporary ``.part`` file and renames on success, so a
    partial download is never mistaken for a complete one on a later run.

    Args:
        session: Session used for the request.
        url: Source URL.
        destination: Target path.
        overwrite: Re-download even if ``destination`` already exists.

    Returns:
        ``True`` if a file was downloaded, ``False`` if skipped or failed.
    """
    if destination.exists() and not overwrite:
        logger.info("Skipping %s (already present)", destination.name)
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    handle.write(chunk)
    except requests.RequestException as exc:
        logger.error("Failed to download %s: %s", url, exc)
        temp_path.unlink(missing_ok=True)
        return False

    temp_path.rename(destination)
    size_mb = destination.stat().st_size / 1e6
    logger.info("Downloaded %s (%.1f MB)", destination.name, size_mb)
    return True


# --------------------------------------------------------------------------- #
# Source-specific downloads
# --------------------------------------------------------------------------- #


def download_taxi_months(
    session: requests.Session,
    start: str,
    end: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Download monthly TLC trip-record PARQUET files for the given range."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for year, month in month_range(start, end):
        filename = f"{TAXI_COLOUR}_tripdata_{year}-{month:02d}.parquet"
        download_file(
            session,
            f"{TLC_BASE_URL}/{filename}",
            output_dir / filename,
            overwrite=overwrite,
        )


def download_taxi_zones(
    session: requests.Session,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Download the taxi zone lookup table and the zone shapefile archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(
        session,
        f"{TLC_MISC_URL}/taxi_zone_lookup.csv",
        output_dir / "taxi_zone_lookup.csv",
        overwrite=overwrite,
    )

    archive_path = output_dir / "taxi_zones.zip"
    downloaded = download_file(
        session,
        f"{TLC_MISC_URL}/taxi_zones.zip",
        archive_path,
        overwrite=overwrite,
    )

    shapefile_dir = output_dir / "taxi_zones"
    if downloaded or not shapefile_dir.exists():
        shapefile_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(shapefile_dir)
        logger.info("Extracted shapefile to %s", shapefile_dir)


def filter_bts_month(csv_path: Path, parquet_path: Path) -> None:
    """Filter one BTS monthly extract to NYC airports and write PARQUET.

    Reads only the columns in :data:`BTS_COLUMNS`, retains flights whose
    origin or destination is an NYC airport, and writes the result. Reducing
    the file at ingest avoids carrying ~250 MB per month of irrelevant
    domestic flights through the rest of the pipeline.

    Args:
        csv_path: Extracted BTS CSV for a single month.
        parquet_path: Destination PARQUET path.
    """
    frame = pd.read_csv(
        csv_path,
        usecols=lambda column: column in BTS_COLUMNS,
        low_memory=False,
    )
    total_rows = len(frame)

    frame = frame[
        frame["Origin"].isin(NYC_AIRPORTS) | frame["Dest"].isin(NYC_AIRPORTS)
    ].copy()

    frame.to_parquet(parquet_path, index=False)
    logger.info(
        "  %s: %s of %s rows retained (%.1f%%)",
        parquet_path.name,
        f"{len(frame):,}",
        f"{total_rows:,}",
        100 * len(frame) / total_rows if total_rows else 0.0,
    )


def download_bts_months(
    session: requests.Session,
    start: str,
    end: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Download, filter, and store BTS on-time performance for a month range."""
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_tmp"

    for year, month in month_range(start, end):
        parquet_path = output_dir / f"bts_ontime_{year}-{month:02d}.parquet"
        if parquet_path.exists() and not overwrite:
            logger.info("Skipping %s (already present)", parquet_path.name)
            continue

        archive_path = work_dir / f"bts_{year}_{month}.zip"
        url = f"{BTS_BASE_URL}_{year}_{month}.zip"
        if not download_file(session, url, archive_path, overwrite=True):
            logger.warning("Could not retrieve BTS data for %d-%02d", year, month)
            continue

        try:
            with zipfile.ZipFile(archive_path) as archive:
                csv_names = [n for n in archive.namelist() if n.endswith(".csv")]
                if not csv_names:
                    logger.error("No CSV inside %s", archive_path.name)
                    continue
                archive.extract(csv_names[0], work_dir)
                filter_bts_month(work_dir / csv_names[0], parquet_path)
        except zipfile.BadZipFile:
            logger.error(
                "%s is not a valid archive; transtats may have served an "
                "error page. Inspect the file and retry.",
                archive_path.name,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download TLC taxi and BTS flight data into data/landing.",
    )
    parser.add_argument(
        "--start", default="2023-01", help="First month, YYYY-MM (default: 2023-01)."
    )
    parser.add_argument(
        "--end", default="2024-06", help="Last month, YYYY-MM (default: 2024-06)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/landing"),
        help="Landing directory (default: data/landing).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-download existing files."
    )
    parser.add_argument(
        "--skip-taxi", action="store_true", help="Skip TLC trip records."
    )
    parser.add_argument(
        "--skip-bts", action="store_true", help="Skip BTS flight data."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the download pipeline. Returns a process exit code."""
    args = parse_args(argv)
    session = build_session()

    logger.info("Retrieving %s to %s into %s", args.start, args.end, args.output)

    try:
        if not args.skip_taxi:
            logger.info("--- TLC trip records ---")
            download_taxi_months(
                session,
                args.start,
                args.end,
                args.output / "tlc",
                overwrite=args.overwrite,
            )
            logger.info("--- Taxi zones ---")
            download_taxi_zones(
                session, args.output / "tlc", overwrite=args.overwrite
            )

        if not args.skip_bts:
            logger.info("--- BTS on-time performance ---")
            download_bts_months(
                session,
                args.start,
                args.end,
                args.output / "bts",
                overwrite=args.overwrite,
            )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Done. Landing directory: %s", args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
