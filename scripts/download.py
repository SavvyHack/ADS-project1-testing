"""Download raw project data into ``data/landing``.

Pulls four sources in a single pass:

1. NYC TLC Yellow Taxi trip records (monthly PARQUET).
2. TLC taxi zone lookup table and zone shapefile.
3. BTS Reporting Carrier On-Time Performance (monthly ZIP), filtered down to
   flights touching the NYC airports before being written to disk. One extra
   month is fetched ahead of ``--start``; see ``bts_start_month``.
4. Meteostat hourly weather observations for each NYC airport and for Central
   Park, retrieved via the Meteostat bulk interface rather than over HTTP.

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
    python scripts/download.py --skip-taxi --skip-bts   # weather only
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from meteostat import Hourly, Stations
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

#: Sites for which hourly weather observations are retrieved, as
#: (latitude, longitude). The three airports are the sites that matter here: a
#: driver queueing at JFK is exposed to the weather at JFK, not to the citywide
#: average, and the same system drives both the arrival delays and the
#: passenger's willingness to wait for a cab. Central Park is retained as a
#: citywide reference series for the exploratory analysis.
WEATHER_SITES = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "NYC": (40.7794, -73.9692),  # Central Park
}

#: Meteostat columns retained. ``dwpt`` (dew point) is dropped as redundant
#: given temperature and humidity, and ``tsun`` (sunshine minutes) is
#: essentially unpopulated for US stations.
WEATHER_COLUMNS = [
    "temp",   # air temperature, degrees Celsius
    "rhum",   # relative humidity, percent
    "prcp",   # precipitation over the hour, mm
    "snow",   # snow depth, mm -- see caveat in download_weather_sites
    "wspd",   # average wind speed, km/h
    "wpgt",   # peak wind gust, km/h
    "pres",   # sea-level air pressure, hPa
    "coco",   # Meteostat weather condition code
]

#: Every timestamp in this project is New York wall-clock time.
LOCAL_TZ = "America/New_York"

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


def bts_start_month(start: str) -> str:
    """Return the month preceding ``start``.

    BTS records ``FlightDate`` as the **departure** date, so a flight that
    leaves the west coast late on the last day of the preceding month lands in
    the small hours of the first day of the study window. Those arrivals belong
    to the window but live in the previous month's file, and without them the
    first morning of the data is short by roughly ten percent with nothing to
    indicate it. One extra month is therefore always fetched; the preprocessing
    notebook filters on arrival date and discards the remainder.

    Args:
        start: First month of the study window, ``YYYY-MM``.

    Returns:
        The preceding month, ``YYYY-MM``.

    Raises:
        ValueError: If ``start`` fails to parse.
    """
    try:
        year, month = (int(part) for part in start.split("-"))
    except ValueError as exc:
        raise ValueError("Months must be formatted as YYYY-MM.") from exc

    return f"{year - 1}-12" if month == 1 else f"{year}-{month - 1:02d}"


def window_bounds(start: str, end: str) -> tuple[datetime, datetime]:
    """Return the half-open datetime bounds of a ``YYYY-MM`` month range.

    Args:
        start: First month, ``YYYY-MM``.
        end: Last month, ``YYYY-MM``, inclusive.

    Returns:
        ``(begin, finish)`` where ``begin`` is midnight on the first day of
        ``start`` and ``finish`` is midnight on the first day of the month
        after ``end``. Both are naive and denote New York local time.
    """
    months = list(month_range(start, end))
    first_year, first_month = months[0]
    last_year, last_month = months[-1]

    begin = datetime(first_year, first_month, 1)
    if last_month == 12:
        finish = datetime(last_year + 1, 1, 1)
    else:
        finish = datetime(last_year, last_month + 1, 1)
    return begin, finish


def resolve_station(latitude: float, longitude: float) -> pd.Series:
    """Find the Meteostat station nearest a coordinate.

    Station identifiers are resolved from coordinates rather than hard-coded,
    so the station actually used is discovered and logged at run time rather
    than asserted. The identifier and its distance from the site are written to
    a manifest so the report can cite the specific station.

    Args:
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.

    Returns:
        The station record, whose index label is the Meteostat station id.

    Raises:
        ValueError: If no station is found near the coordinate.
    """
    stations = Stations().nearby(latitude, longitude).fetch(1)
    if stations.empty:
        raise ValueError(f"No weather station found near ({latitude}, {longitude})")
    return stations.iloc[0]


def download_weather_sites(
    start: str,
    end: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Retrieve hourly weather observations for every site in WEATHER_SITES.

    Four decisions here are deliberate and belong in the report's
    preprocessing section.

    **Observations only.** Meteostat's ``model`` parameter defaults to True,
    which backfills gaps in the observation record with numerical weather model
    output. Those are not measurements. It is set to False, so every retained
    row is a real observation and a gap appears as a null that can be counted
    rather than as a plausible number of unknown provenance. Per-column null
    rates are logged for exactly this reason.

    **Explicit timezone conversion.** Meteostat returns a naive UTC index.
    Rather than passing its ``timezone`` argument and inheriting its DST
    handling, the index is localised to UTC and converted here, matching the
    convention used for the TLC and BTS data. The request is padded by a day at
    each end so conversion cannot clip the first or last local hour.

    **The duplicated fall-back hour.** On the date the clocks go back, 01:00
    occurs twice, producing two rows on the same ``(obs_date, obs_hour)`` key.
    Left alone this silently duplicates rows when joined to the taxi table, so
    the pair is collapsed by averaging.

    **Snow depth is usually empty.** ``snow`` is snow *depth*, which US METAR
    stations rarely report hourly. Expect it to be almost entirely null and
    derive snowfall from the ``coco`` condition code instead, after inspecting
    which codes actually occur in the retrieved window.

    Args:
        start: First month, ``YYYY-MM``.
        end: Last month, ``YYYY-MM``, inclusive.
        output_dir: Destination directory for the per-site PARQUET files.
        overwrite: Re-download sites whose output already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    begin, finish = window_bounds(start, end)
    manifest = []

    for code, (latitude, longitude) in WEATHER_SITES.items():
        parquet_path = output_dir / f"weather_{code.lower()}.parquet"
        if parquet_path.exists() and not overwrite:
            logger.info("Skipping %s (already present)", parquet_path.name)
            continue

        station = resolve_station(latitude, longitude)
        logger.info(
            "%s -> station %s (%s), %.1f km away",
            code,
            station.name,
            station["name"],
            station["distance"] / 1000,
        )

        # Pad the UTC request by a day either side. New York is UTC-4 or -5, so
        # an unpadded request loses hours at both boundaries on conversion.
        frame = Hourly(
            station.name,
            begin - timedelta(days=1),
            finish + timedelta(days=1),
            model=False,
        ).fetch()

        if frame.empty:
            logger.warning("No observations returned for %s", code)
            continue

        frame = frame.reindex(columns=WEATHER_COLUMNS)
        frame.index = frame.index.tz_localize("UTC").tz_convert(LOCAL_TZ)

        # Clip to the study window in local time, then split the local
        # wall-clock timestamp into the key used by every other table.
        frame = frame[
            (frame.index >= begin.isoformat()) & (frame.index < finish.isoformat())
        ]
        frame = frame.reset_index(names="observed_at")
        frame["site"] = code
        frame["obs_date"] = frame["observed_at"].dt.date
        frame["obs_hour"] = frame["observed_at"].dt.hour
        frame["observed_at"] = frame["observed_at"].dt.tz_localize(None)

        duplicates = int(frame.duplicated(subset=["obs_date", "obs_hour"]).sum())
        if duplicates:
            logger.info("  collapsing %d duplicated DST hour(s)", duplicates)
            frame = frame.groupby(
                ["site", "obs_date", "obs_hour"], as_index=False
            ).agg({**{column: "mean" for column in WEATHER_COLUMNS},
                   "observed_at": "min"})

        frame.to_parquet(parquet_path, index=False)

        expected_hours = int((finish - begin).total_seconds() // 3600)
        null_rates = {
            column: round(float(frame[column].isna().mean()), 4)
            for column in WEATHER_COLUMNS
        }
        logger.info(
            "  %s: %s of %s hours present; null rates %s",
            parquet_path.name,
            f"{len(frame):,}",
            f"{expected_hours:,}",
            null_rates,
        )

        manifest.append(
            {
                "site": code,
                "station_id": station.name,
                "station_name": station["name"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "distance_km": round(station["distance"] / 1000, 2),
                "hours_retrieved": len(frame),
                "hours_expected": expected_hours,
            }
        )

    if manifest:
        manifest_path = output_dir / "weather_stations.csv"
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
        logger.info("Station manifest written to %s", manifest_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download TLC taxi, BTS flight, and weather data into data/landing."
        ),
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
    parser.add_argument(
        "--skip-weather", action="store_true", help="Skip weather observations."
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
            # One month earlier than the taxi data, deliberately. See
            # `bts_start_month`.
            flight_start = bts_start_month(args.start)
            logger.info(
                "--- BTS on-time performance (from %s, one month before the "
                "window, to recover overnight arrivals) ---",
                flight_start,
            )
            download_bts_months(
                session,
                flight_start,
                args.end,
                args.output / "bts",
                overwrite=args.overwrite,
            )

        if not args.skip_weather:
            logger.info("--- Hourly weather observations ---")
            download_weather_sites(
                args.start,
                args.end,
                args.output / "weather",
                overwrite=args.overwrite,
            )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Done. Landing directory: %s", args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
