# NYC Yellow Taxi Airport Demand — MAST30034 Project 1

**Student:** Tavish Balyan | **Student ID:** 1615873

Research goal.

- **Dataset:** NYC TLC Yellow Taxi trip records, January 2023 – June 2024
  (18 monthly files, 58,642,319 trips)
- **External dataset:** [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ),
  filtered to flights touching JFK, LGA, and EWR (19 monthly files, 1,349,639 flights).
  Nineteen, not eighteen: BTS records the *departure* date, so arrivals in the small
  hours of 1 January 2023 sit in the December 2022 file. `download.py` fetches one month
  ahead of the window automatically and `2b` discards whatever lands outside it.
- **Report:** `report/main.pdf`

---

## Installation

Requires Python 3.11+ and Java 17 (PySpark 3.5 does not support Java 21+).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Downloading the data

```bash
python scripts/download.py --start 2023-01 --end 2024-06
```

Writes roughly 950 MB to `data/landing/` in about 10 minutes — 935 MB of trip records
and 12 MB of filtered flight data across 19 months. The range above is the default; results
reported in `report/main.pdf` use exactly this range. Downloads are idempotent —
if the run is interrupted, re-run the same command and completed files are
skipped. `--help` lists all options.

Raw data is not committed to this repository.

### Troubleshooting

The BTS endpoint (`transtats.bts.gov`) is the least reliable component of the
pipeline and accounts for essentially all download failures observed. In rough
order of likelihood:

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `BadZipFile` raised on extract | Endpoint returned an HTML error page with HTTP 200 | Transient. Re-run the same command. |
| Hanging or timed-out connection | Large archive over a slow link | Read timeout is 300 s with 5 retries and exponential backoff. Re-run. |
| HTTP 403 | Default urllib user agent rejected | Already handled — the script sends a browser user agent. |
| `SSLError` on campus or corporate networks | TLS interception | Add `verify=False` to the session in `download.py`. Last resort only. |
| Repeated 404 for a recent month | Month not yet published | BTS lags roughly 2–3 months behind the current date. |

If BTS remains unavailable, the taxi data can be fetched on its own with
`--skip-bts`, and the flight data added later in the same command.

Each BTS monthly archive expands to roughly 250 MB of CSV covering all domestic
US flights. Only flights touching JFK, LGA, or EWR are retained, reducing the
19-month feed to 1,349,639 rows; the script prints per-month retention as it runs.

## Running the pipeline

Execute in order. Each notebook reads the previous stage's output.

| Order | Notebook | Purpose | Reads | Writes |
|-------|----------|---------|-------|--------|
| 1 | `notebooks/2a_preprocess_taxi.ipynb` | Business-rule filtering of all 58.6M trips, then aggregation of the JFK and LGA subset to airport-hours | `data/landing/tlc/` | `data/raw/trips_clean.parquet`, `data/curated/taxi_airport_hourly.parquet` |
| 2 | `notebooks/2b_preprocess_flights.ipynb` | New York local arrival hour, then aggregation to the same airport-hour grid | `data/landing/bts/` | `data/curated/flights_hourly.parquet`, `data/curated/flights_hourly_ewr.parquet` |
| 3 | `notebooks/2c_join_features.ipynb` | Join on the airport-hour key, temporal, holiday, and lag features, train/test split | `data/curated/` | `data/curated/model_table.parquet` |
| 4 | `notebooks/3_analysis.ipynb` | Distributions, outliers, geospatial visualisation | `data/raw/`, `data/curated/` | `plots/` |
| 5 | `notebooks/4_modelling.ipynb` | Poisson GLM and gradient-boosted trees | `data/curated/` | `models/`, `plots/` |

All four curated tables share one key: `(date, hour, airport)` for JFK and LGA, 547 days
× 24 hours × 2 airports = **26,256 rows**. Table A carries it as
`(pickup_date, pickup_hour, airport)`; notebook 2c renames those two columns on load. Any
join that changes the row count is a bug, and 2c asserts the count after each one.

Notebooks import shared helpers from `scripts/spark_utils.py` via
`sys.path.append("../scripts")`.

## Repository structure

```
├── data/                  # gitignored
│   ├── landing/           # untouched downloads
│   ├── raw/               # after filtering and type coercion
│   └── curated/           # model-ready, joined
├── notebooks/             # numbered, run in order
├── scripts/
│   ├── download.py        # data acquisition
│   └── spark_utils.py     # session config, schema normalisation, loaders
├── plots/                 # figures used in the report
├── models/                # fitted model artefacts
├── report/                # LaTeX source and compiled PDF
├── requirements.txt
└── README.md
```

## Data dictionary

| Source | Reference |
|--------|-----------|
| TLC Yellow Taxi | [Data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) |
| TLC trip records overview | [User guide](https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf) |
| TLC taxi zones | [Lookup table](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) |
| BTS on-time performance | [Field definitions](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) |

Relevant taxi zones, verified against the lookup table: **132** JFK Airport,
**138** LaGuardia Airport, **1** Newark Airport (EWR).

## Source data quirks

Inconsistencies identified during ingest and handled in preprocessing. Each is
addressed in `scripts/spark_utils.py` or `notebooks/1_preprocessing.ipynb`.

**TLC trip records**

- The January 2023 export names the airport surcharge `airport_fee` and stores
  `VendorID`, `PULocationID`, and `DOLocationID` as 64-bit integers. All
  seventeen later months use `Airport_fee` and 32-bit integers. Unioning the
  raw frames therefore yields two sparsely populated fee columns. All column
  names are lower-cased and integer widths fixed before union, and
  `unionByName` is used so column ordering cannot misalign.
- Timestamps carry no timezone and are local New York wall-clock. The Spark
  session pins `spark.sql.session.timeZone` to `America/New_York` accordingly.

**BTS on-time performance**

- Times are stored as `hhmm` integers (`905` denotes 09:05), and `2400` denotes
  midnight, which must be mapped to hour 0 of the following day.
- Times are local to the airport concerned. For arrival features only rows with
  `Dest` in {JFK, LGA} are used, so that `CRSArrTime` and `ArrTime` are New York
  local; departure features use `Origin` with `CRSDepTime`.
- `FlightDate` records the **departure** date. Overnight flights where
  `CRSArrTime < CRSDepTime` arrive on the following calendar day and are
  shifted forward before being bucketed into an hour.
- `ArrTime` is null for 1.42% of rows, corresponding to cancelled flights.
- Each month is written by pandas, which infers dtypes per file. A month with no
  cancellations stores `Cancelled` as an integer where every other month stores a
  float, and `ArrTime` and `ArrDelayMinutes` drift the same way. `normalise_bts_schema`
  fixes the widths before union.
- The three clock corrections in `add_arrival_key` are **order-dependent**: `2400 → 0`
  must precede the overnight date shift, or a red-eye scheduled to arrive at `2400`
  is recorded as landing at midnight at the *start* of its departure day. See notebook
  2b, Step 2.
- Cancelled flights are retained in the scheduled series, since a cancellation is
  visible on the arrivals board, and excluded from the realised series along with
  diversions.
- `FlightDate` is the departure date, so arrivals in the small hours of 1 January 2023
  are recorded in the December 2022 file. `download.py` fetches one month ahead of the
  window for this reason and notebook 2b keeps the 24 arrivals that land inside it,
  discarding the rest of that month. 2b asserts the preceding month is present, so the
  recovery cannot silently fail to happen.
- Two rows imply a 22-hour domestic block time, both Republic Airways regional legs into
  LaGuardia whose scheduled arrival precedes its scheduled departure by two hours. At
  least one clock field is corrupt in each, and the overnight correction assigns them to
  the wrong day, so they are removed.

## Preprocessing summary

Record counts at each stage, mirroring the table in the report.

**TLC trip records** (`data/curated/preprocessing_counts_taxi.csv`)

| Step | Rows | Removed | Justification |
|------|------|---------|---------------|
| Raw TLC ingest | 58,642,319 | — | 18 monthly files, Jan 2023 – Jun 2024 |
| Pickup within study window | 58,642,192 | 127 | Meter and upload errors timestamped outside the file's own month |
| Drop-off after pickup | 58,620,446 | 21,746 | A trip cannot end before it starts |
| Duration between 1 min and 6 h | 57,922,347 | 698,099 | Sub-minute records are meter mis-taps; multi-hour records are meters left running |
| Positive trip distance | 57,243,639 | 678,708 | Zero distance means no journey took place |
| Positive fare and total | 56,642,200 | 601,439 | Negative amounts are voided transactions and chargebacks |
| Metered fare at or above initial charge | 56,592,091 | 50,109 | Below the published initial charge on Rate Code 1 |
| Implied speed below 80 mph | 56,588,396 | 3,695 | Not attainable across the five boroughs; corrupt odometer or timestamp |
| Documented rate code | 53,451,022 | 3,137,374 | Data dictionary defines codes 1–6; code 99 is undocumented |
| Documented payment type | 53,451,022 | 0 | Data dictionary defines codes 1–6 |

Of the cleaned 53,451,022 trips, **4,637,237** (8.68%) are airport pickups: 2,746,141 at
JFK and 1,891,096 at LaGuardia. These aggregate to 25,009 non-empty airport-hours, which
the manufactured spine completes to 26,256; 1,242 of the remainder are genuine
zero-pickup hours and 6 are daylight-saving artefacts, flagged rather than deleted.

The zone filter is validated independently of the lookup table: 96.19% of the 2,019,422
Rate Code 2 (JFK flat fare) trips touch zone 132 at one end, and the rate code is entered
at the meter rather than derived from GPS.

**BTS on-time performance** (`data/curated/preprocessing_counts_flights.csv`)

| Step | Rows | Removed | Justification |
|------|------|---------|---------------|
| Raw BTS ingest | 1,349,639 | — | 19 monthly files: the 18 in the window plus December 2022 |
| Arrival at JFK or LGA | 461,205 | 888,434 | Only arrivals generate pickups, and the filter guarantees the clock fields are New York local |
| Well-formed scheduled clock fields | 461,205 | 0 | `hhmm` readings outside [0, 2400] or with a minute component of 60+ cannot be bucketed. No record failed |
| Plausible implied block time | 461,203 | 2 | Two rows imply a 22-hour domestic leg; at least one clock field is corrupt and the overnight correction assigns them to the wrong day |
| Scheduled arrival within study window | 436,754 | 24,449 | Applied after the overnight shift, so arrival date rather than departure date decides. Removes December 2022 except the arrivals that roll into 1 January, and the 49 June 2024 arrivals that roll past 1 July |

Of the in-window scheduled arrivals, 424,670 (97.2%) actually landed; the remainder were
cancelled or diverted. These aggregate onto the same 26,256-row grid as the taxi table.
JFK schedules nothing to land in 1,910 of its hours and LaGuardia in 3,432 — the latter
is the overnight curfew, at 6.3 hours a day.

## Notes and known limitations

- The two busiest scheduled hours differ by airport: JFK peaks at 14:00 and 07:00, LGA at
  13:00 and 17:00. LaGuardia's `share_longhaul` is 0.034 against JFK's 0.385, which is the
  perimeter rule appearing in the data without being told to.
- BTS covers domestic flights only; international arrivals at JFK are not
  represented. Discussed in the report.