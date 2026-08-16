"""Shared Spark helpers for loading and normalising TLC trip records.

Imported by the notebooks in ``notebooks/``. Keeping session configuration and
schema normalisation here means every notebook loads the data identically, and
the logic can be linted and reviewed as ordinary source rather than as
notebook cells.
"""

from __future__ import annotations

import glob
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

#: Columns cast to 32-bit integers during normalisation.
_INT_COLUMNS = ("vendorid", "pulocationid", "dolocationid", "payment_type",
                "passenger_count", "ratecodeid")

#: Columns cast to double during normalisation.
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
