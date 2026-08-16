"""Shared Spark helpers for loading and normalising the project's datasets.

Imported by the notebooks in ``notebooks/``. Keeping session configuration and
schema normalisation here means every notebook loads the data identically, and
the logic can be linted and reviewed as ordinary source rather than as
notebook cells.

The module covers the TLC trip records and the BTS on-time performance feed,
plus the count-waterfall and hourly-spine helpers used by both preprocessing
notebooks.
"""

from __future__ import annotations

import glob
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

#: TLC columns cast to 32-bit integers during normalisation.
_INT_COLUMNS = ("vendorid", "pulocationid", "dolocationid", "payment_type",
                "passenger_count", "ratecodeid")

#: TLC columns cast to double during normalisation.
_DOUBLE_COLUMNS = ("airport_fee", "congestion_surcharge")

#: Taxi zone identifiers, verified against ``taxi_zone_lookup.csv``.
JFK_ZONE = 132
LGA_ZONE = 138
EWR_ZONE = 1


def create_spark_session(
    app_name: str = "MAST30034",
    driver_memory: str = "3g",
    shuffle_partitions: int = 32,
) -> SparkSession:
    """Create a locally configured Spark session.

    The default driver memory of 1 GB is insufficient for eighteen months of
    trip records, and the default of 200 shuffle partitions is wasteful on a
    single machine. Both are raised here so notebooks do not need to repeat
    the configuration.

    Args:
        app_name: Name shown in the Spark UI.
        driver_memory: JVM heap for the driver. Keep below the memory
            available to the WSL2 VM.
        shuffle_partitions: Partition count for shuffle operations.

    Returns:
        An active :class:`SparkSession`.
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.session.timeZone", "America/New_York")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def normalise_schema(df: DataFrame) -> DataFrame:
    """Standardise column casing and column types across monthly exports.

    TLC's PARQUET exports are not internally consistent. The January 2023 file
    names the airport surcharge ``airport_fee`` and stores location and vendor
    identifiers as 64-bit integers; every subsequent month through June 2024
    uses ``Airport_fee`` with 32-bit identifiers. Unioning the raw frames
    therefore either fails or produces two sparsely populated fee columns.

    Lower-casing every column name and casting to fixed widths makes the
    monthly frames union-compatible. Columns absent from a given month are
    left alone rather than created, so callers can detect genuine gaps.

    Args:
        df: A single month of trip records as read from PARQUET.

    Returns:
        The frame with normalised column names and types.
    """
    df = df.toDF(*[column.lower() for column in df.columns])

    for column in _INT_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("int"))

    for column in _DOUBLE_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("double"))

    return df


def load_trips(spark: SparkSession, landing_dir: Path | str) -> DataFrame:
    """Load and union every monthly trip-record file in ``landing_dir``.

    Each month is normalised before union, and ``unionByName`` is used rather
    than ``union`` so that any difference in column ordering cannot silently
    misalign columns.

    Args:
        spark: Active Spark session.
        landing_dir: Directory holding ``yellow_tripdata_*.parquet``.

    Returns:
        A single frame spanning all available months.

    Raises:
        FileNotFoundError: If no trip-record files are present.
    """
    pattern = str(Path(landing_dir) / "yellow_tripdata_*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trip records matched {pattern}")

    frames = [normalise_schema(spark.read.parquet(path)) for path in paths]
    return reduce(DataFrame.unionByName, frames)


#: BTS columns cast to 32-bit integers during normalisation. The four clock
#: fields are ``hhmm`` integers, not times, and are handled as integers all the
#: way through to the hour derivation in notebook 2b.
_BTS_INT_COLUMNS = (
    "crsdeptime",
    "deptime",
    "crsarrtime",
    "arrtime",
    "flight_number_reporting_airline",
)

#: BTS columns cast to double during normalisation.
_BTS_DOUBLE_COLUMNS = (
    "depdelayminutes",
    "arrdelayminutes",
    "cancelled",
    "diverted",
    "distance",
)

#: Airports at which a yellow taxi may pick up. Newark is deliberately absent:
#: it lies outside the TLC pickup zones and enters this study only as a
#: flight-side context variable.
ARRIVAL_AIRPORTS = ("JFK", "LGA")
EWR_AIRPORT = "EWR"


def normalise_bts_schema(df: DataFrame) -> DataFrame:
    """Standardise column casing and column types across monthly BTS exports.

    ``download.py`` writes each BTS month with pandas, which infers dtypes per
    file. A month containing no cancellations stores ``Cancelled`` as an
    integer while every other month stores it as a float, and ``ArrTime`` and
    ``ArrDelayMinutes`` drift the same way because they are null for cancelled
    flights. Unioning the raw frames therefore fails on type mismatch.

    Column names are lower-cased so the flight and taxi tables follow one
    convention. Columns absent from a given month are left alone rather than
    created, so callers can detect genuine gaps.

    Args:
        df: A single month of on-time performance records as read from PARQUET.

    Returns:
        The frame with normalised column names and types.
    """
    df = df.toDF(*[column.lower() for column in df.columns])

    for column in _BTS_INT_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("int"))

    for column in _BTS_DOUBLE_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("double"))

    return df


def load_flights(spark: SparkSession, landing_dir: Path | str) -> DataFrame:
    """Load and union every monthly on-time performance file in ``landing_dir``.

    Mirrors :func:`load_trips`: each month is normalised before union, and
    ``unionByName`` is used rather than ``union`` so that any difference in
    column ordering cannot silently misalign columns.

    Args:
        spark: Active Spark session.
        landing_dir: Directory holding ``bts_ontime_*.parquet``.

    Returns:
        A single frame spanning all available months.

    Raises:
        FileNotFoundError: If no on-time performance files are present.
    """
    pattern = str(Path(landing_dir) / "bts_ontime_*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No flight records matched {pattern}")

    frames = [normalise_bts_schema(spark.read.parquet(path)) for path in paths]
    return reduce(DataFrame.unionByName, frames)


def count_waterfall(df: DataFrame, filters: list) -> list:
    """Compute the row count remaining after each filter, in a single pass.

    Calling ``.count()`` after each filter would trigger one full scan per
    filter. Instead the cumulative conjunction of the predicates is evaluated
    in a single pass: summing ``predicate_1``, then ``predicate_1 AND
    predicate_2``, and so on, gives exactly the sequential post-filter counts.

    ``F.when(condition, 1).otherwise(0)`` is used rather than casting the
    boolean, so that a null predicate counts as an exclusion. This matches the
    behaviour of ``.where()``, which also drops null predicates.

    Args:
        df: The unfiltered DataFrame.
        filters: Ordered list of ``(label, predicate)`` pairs.

    Returns:
        List of ``(label, rows_remaining, rows_removed)`` tuples, preceded by
        an entry for the unfiltered frame.
    """
    cumulative = F.lit(True)
    aggregations = [F.count("*").alias("stage_0")]
    for i, (_, predicate) in enumerate(filters, start=1):
        cumulative = cumulative & predicate
        aggregations.append(
            F.sum(F.when(cumulative, 1).otherwise(0)).alias(f"stage_{i}")
        )

    row = df.agg(*aggregations).collect()[0]

    waterfall = [("Raw ingest", row["stage_0"], None)]
    for i, (label, _) in enumerate(filters, start=1):
        remaining = row[f"stage_{i}"]
        waterfall.append((label, remaining, row[f"stage_{i - 1}"] - remaining))
    return waterfall


def print_waterfall(waterfall: list, label_width: int = 42) -> None:
    """Print a count waterfall as an aligned table.

    Args:
        waterfall: Output of :func:`count_waterfall`.
        label_width: Column width for the step labels.
    """
    raw_total = waterfall[0][1]
    for label, remaining, removed in waterfall:
        removed_str = "—" if removed is None else f"{removed:,}"
        share = 100 * remaining / raw_total if raw_total else 0.0
        print(
            f"{label:<{label_width}} {remaining:>12,}  "
            f"removed {removed_str:>10}  ({share:6.2f}% of raw)"
        )


def hour_spine(
    spark: SparkSession,
    start: str,
    end: str,
    airports: tuple | list | None = None,
    date_col: str = "date",
    hour_col: str = "hour",
) -> DataFrame:
    """Build a complete grid of every hour in a half-open date window.

    An hour in which nothing happened produces no group in an aggregation and
    is therefore absent from it. Left-joining the aggregation onto a
    manufactured spine restores those hours, which is essential when the
    quantity being aggregated is a count: a model fitted only to hours that
    had events will systematically overpredict quiet periods.

    Args:
        spark: Active Spark session.
        start: First date in the window, ``YYYY-MM-DD``, inclusive.
        end: Day after the last date in the window, ``YYYY-MM-DD``, exclusive.
        airports: Airport codes to cross-join, or ``None`` for a grid with no
            airport dimension.
        date_col: Name for the date column.
        hour_col: Name for the hour column.

    Returns:
        One row per ``(date, hour)`` or ``(date, hour, airport)``.
    """
    spine = spark.sql(
        f"SELECT explode(sequence(to_date('{start}'), "
        f"date_sub(to_date('{end}'), 1), interval 1 day)) AS {date_col}"
    ).crossJoin(
        spark.range(24).select(F.col("id").cast("int").alias(hour_col))
    )

    if airports is not None:
        spine = spine.crossJoin(
            spark.createDataFrame([(code,) for code in airports], ["airport"])
        )

    return spine