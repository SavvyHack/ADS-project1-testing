# NYC Yellow Taxi Airport Demand — MAST30034 Project 1

**Student:** Tavish Balyan | **Student ID:** 1615873

**Research goal.** Where and when should a New York yellow taxi driver join an airport
queue? The project models two quantities per airport-hour — how many pickups occur, and
what each is worth — and combines them into expected revenue per hour of queueing at JFK
and LaGuardia, using the published arrivals schedule and observed weather as the
predictors a driver could actually act on in advance.

- **Dataset:** NYC TLC Yellow Taxi trip records, January 2023 – June 2024
  (18 monthly files, 58,642,319 trips)
- **External dataset 1:** [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ),
  filtered to flights touching JFK, LGA, and EWR (19 monthly files, 1,349,639 flights).
  Nineteen, not eighteen: BTS records the *departure* date, so arrivals in the small
  hours of 1 January 2023 sit in the December 2022 file. `download.py` fetches one month
  ahead of the window automatically and `2b` discards whatever lands outside it.
- **External dataset 2:** [Meteostat](https://meteostat.net/) hourly weather observations
  for JFK, LGA, EWR, and Central Park (4 sites × 13,128 hours). Retrieved through the
  [Meteostat Python library](https://dev.meteostat.net/python/) with `model=False`, so
  every retained row is a station observation rather than numerical-model output. The
  stations actually resolved, their distance from each airport, and the hours returned are
  written to `data/landing/weather/weather_stations.csv` at download time, for citation in
  the report.
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

Writes roughly 950 MB to `data/landing/` in about 10 minutes — 935 MB of trip records,
12 MB of filtered flight data across 19 months, the taxi zone lookup table and shapefile,
and a few MB of hourly weather; the zone files land in `data/landing/tlc/`. The range above
is the default, and the results in `report/main.pdf` use exactly this range.

Downloads are idempotent — if the run is interrupted, re-run the same command and completed
files are skipped. Individual sources can be skipped with `--skip-taxi` (which also covers
the zone lookup and shapefile), `--skip-bts`, and `--skip-weather`; `--overwrite` forces a
re-fetch. `--help` lists all options.

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
| 1 | `notebooks/2a_preprocess_taxi.ipynb` | Business-rule filtering of all 58.6M trips, then aggregation of the JFK and LGA subset to airport-hours | `data/landing/tlc/` | `data/raw/trips_clean.parquet`, `data/curated/taxi_airport_hourly.parquet`, `preprocessing_counts_taxi.csv`, `shapes_taxi.json` |
| 2 | `notebooks/2b_preprocess_flights.ipynb` | New York local arrival hour, then aggregation to the same airport-hour grid | `data/landing/bts/` | `data/curated/flights_hourly.parquet`, `flights_hourly_ewr.parquet`, `flight_column_roles.json`, `preprocessing_counts_flights.csv`, `shapes_flights.json` |
| 3 | `notebooks/2c_join_features.ipynb` | Join on the airport-hour key; weather cleaning, temporal, holiday, and lag features; leakage enforcement; train/test split | `data/curated/`, `data/landing/weather/` | `data/curated/model_table.parquet`, `model_table_roles.json`, `shapes_model_table.json` |
| 4 | `notebooks/3_analysis.ipynb` | Distributions, outliers, temporal and weather analysis, geospatial visualisation | `data/raw/`, `data/curated/`, `data/landing/tlc/` | `plots/` (7 figures), `distribution_summary.csv`, `analysis_findings.json` |
| 5 | `notebooks/4_modelling.ipynb` | Poisson GLM for demand, gradient-boosted trees for fare, and their product | `data/curated/model_table.parquet`, `model_table_roles.json` | `models/`, `data/curated/model_predictions.parquet`, `model_results.json`, `plots/` (3 figures) |

All four curated tables share one key: `(date, hour, airport)` for JFK and LGA, 547 days
× 24 hours × 2 airports = **26,256 rows**. Table A carries it as
`(pickup_date, pickup_hour, airport)`; notebook 2c renames those two columns on load. Any
join that changes the row count is a bug, and 2c asserts the count after each one.

Notebooks import shared helpers from `scripts/spark_utils.py` and, from notebook 3
onward, figure styling from `scripts/plot_utils.py`, via `sys.path.append("../scripts")`.
Every figure in the report is drawn at the template's text width through `plot_utils`, so
that one palette, one font size, and one export resolution apply across the report.

## Modelling

Two models with different jobs, combined rather than raced:

| | Model 1 | Model 2 |
|---|---|---|
| Question | How many pickups in this airport-hour? | What is a pickup worth in it? |
| Target | `n_pickups` | `mean_total` |
| Family | Poisson GLM, log link, quasi-Poisson scale, hour interacted with airport | `HistGradientBoostingRegressor` |
| Missing data | Complete design matrix; imputation fitted on the training split | Handled natively, unimputed |
| Read through | Rate ratios and an adjusted hourly demand profile | Permutation importance on the test split |

Their product is expected revenue per airport-hour, which rests on the identity 2c
asserts: `n_pickups × mean_total` reproduces `sum_total_amount` to the cent. The
combination is evaluable on every test hour, including the empty ones where the value
model alone is undefined, against the product of the two seasonal-naive forecasts.

A negative binomial is fitted on the same design as a contrast, with its dispersion
estimated by the standard auxiliary regression, so the demand result can be reported as
robust to the variance assumption rather than conditional on it. Boosting hyper-parameters
are chosen on October–December 2023 and refitted on the full training year: the split is
forward in time at every level, including model selection.

**Market revenue is not a wage.** `n_pickups × mean_total` is what the fleet collectively
earned from a queue, not what one driver takes home; the TLC data records trips, not
waiting taxis, so queue length is unobserved. The defensible use is comparative — which
queue is worth more at the same moment — and the report states the assumption.

## Repository structure

```
├── data/                  # gitignored
│   ├── landing/           # untouched downloads (tlc, bts, weather, zones)
│   ├── raw/               # after filtering and type coercion
│   └── curated/           # model-ready, joined
├── notebooks/             # numbered, run in order
├── scripts/
│   ├── download.py        # data acquisition
│   ├── spark_utils.py     # session config, schema normalisation, loaders
│   └── plot_utils.py      # palette, figure sizing, saving at 300 dpi
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
| Meteostat hourly data | [Format and condition codes](https://dev.meteostat.net/formats.html) |

Relevant taxi zones, verified against the lookup table: **132** JFK Airport,
**138** LaGuardia Airport, **1** Newark Airport (EWR).

## Source data quirks

Inconsistencies identified during ingest and handled in preprocessing. Each is
addressed in `scripts/download.py`, `scripts/spark_utils.py`, or the notebook named.

**TLC trip records**

- The January 2023 export names the airport surcharge `airport_fee` and stores
  `VendorID`, `PULocationID`, and `DOLocationID` as 64-bit integers. All
  seventeen later months use `Airport_fee` and 32-bit integers. Unioning the
  raw frames therefore yields two sparsely populated fee columns. All column
  names are lower-cased and integer widths fixed before union, and
  `unionByName` is used so column ordering cannot misalign.
- Timestamps carry no timezone and are local New York wall-clock. The Spark
  session pins `spark.sql.session.timeZone` to `America/New_York` accordingly.
- Rate code and passenger count are unrecorded in 3,186,927 and 2,793,889 cleaned trips
  respectively (5.63% and 4.93%). The block is not random: it is almost exactly the set of
  Flex Fare trips, and within it the share of null passenger counts and the share of Flex
  Fare payments agree to the digit. These records are retained and flagged rather than
  deleted, because deleting them would push one fare programme's market share into the
  response variable. See notebook 2a, Step 4a.

**BTS on-time performance**

- Times are stored as `hhmm` integers (`905` denotes 09:05), and `2400` denotes
  midnight, which must be mapped to hour 0 of the following day.
- Times are local to the airport concerned. For arrival features only rows with
  `Dest` in {JFK, LGA} are used, so that `CRSArrTime` and `ArrTime` are New York
  local; departure features use `Origin` with `CRSDepTime`.
- `FlightDate` records the **departure** date. Overnight flights where
  `CRSArrTime < CRSDepTime` arrive on the following calendar day and are
  shifted forward before being bucketed into an hour. 4.79% of arrivals are shifted.
- `ArrTime` is null for 10,631 in-window arrivals (2.4%), corresponding to cancellations
  and diversions.
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
  window for this reason and notebook 2b keeps the 24 arrivals that land inside it —
  4.7% of that day's arrivals — discarding the rest of that month. 2b asserts the
  preceding month is present, so the recovery cannot silently fail to happen.
- Two rows imply a 22-hour domestic block time, both Republic Airways regional legs into
  LaGuardia whose scheduled arrival precedes its scheduled departure by two hours. At
  least one clock field is corrupt in each, and the overnight correction assigns them to
  the wrong day, so they are removed.

**Meteostat hourly observations**

- Meteostat writes its index at nanosecond precision, which PySpark 3.5.1 cannot read
  (`Illegal Parquet type: INT64 (TIMESTAMP(NANOS,false))`). `download.py` writes the
  weather files with `coerce_timestamps="us"`, so the parquet carries microsecond
  timestamps. If weather files predating this fix are on disk, regenerate them with
  `python scripts/download.py --skip-taxi --skip-bts --overwrite`.
- `snow` and `wpgt` are **100% null** at all four stations. `snow` is snow *depth*, which
  US METAR stations rarely report hourly, and peak gust is reported no more often. Both
  are dropped in 2c.
- `coco`, the Meteostat condition code, is null in 96.0% of hours at JFK, 96.7% at LGA,
  and 97.4% at EWR — and every populated value across all four stations and eighteen
  months is a **fog** code. It is not a sparse condition series but a fog flag with a
  condition series' schema, so it is dropped and precipitation is derived from the
  measured `prcp` gauge instead.
- `prcp` is missing for 11.9% of hours at JFK, 7.2% at LGA, and 8.3% at EWR, and for the
  entire window at Central Park. 2c resolves it in a fixed order — the site's own gauge,
  then the mean of whichever other airport stations reported that hour, then zero — and
  records which applied in `prcp_source`. After the fallback, 0.7% of airport-hours are
  assumed dry.
- Meteostat returns a naive UTC index. `download.py` localises to UTC and converts to
  `America/New_York` explicitly, padding the request by one day at each end so the
  conversion cannot clip the boundary hours.
- On the date the clocks go back, 01:00 occurs twice; `download.py` collapses the pair by
  averaging so the `(obs_date, obs_hour)` key stays unique. On the two spring-forward
  dates, 02:00 does not exist, which is why each airport station returns 13,126 rows
  rather than 13,128. The spine in 2c restores those two hours and the carry-forward fills
  them, flagged as `weather_imputed`.
- Central Park is missing 459 hours and 28% of its wind readings, so it is used only as an
  independent temperature check on the airport stations, never as a model input.

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
| Metered fare at or above initial charge | 56,641,660 | 540 | Below the $3.00 initial charge on Rate Code 1 (TLC Industry Notice #22-02, in force from 19 December 2022) |
| Implied speed below 80 mph | 56,637,949 | 3,711 | Not attainable across the five boroughs; corrupt odometer or timestamp |
| Documented rate code | 56,637,949 | 0 | Dictionary defines 1–6 and 99 = Null/unknown; nulls and 99 are retained and flagged |
| Documented payment type | 56,637,949 | 0 | Dictionary defines 0–6, where 0 = Flex Fare |

**2,004,370 trips removed in total (3.42%).** The last two filters remove nothing: every
rate code and payment type present in eighteen months is one the dictionary defines. They
are retained in the chain because their absence would be an unstated assumption.

Of the cleaned 56,637,949 trips, **4,653,181** (8.22%) are airport pickups: 2,753,043 at
JFK and 1,900,138 at LaGuardia, generating $354,004,177 in total fares. These aggregate to
25,076 non-empty airport-hours, which the manufactured spine completes to 26,256. Of the
1,180 rows with no pickups, 1,175 are genuine zero-demand hours and 5 are daylight-saving
artefacts, flagged rather than deleted. (The sixth flagged DST row is the fall-back 01:00
hour at JFK, which contains two clock hours' worth of trips rather than none.)

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
| Scheduled arrival within study window | 436,754 | 24,449 | Applied after the overnight shift, so arrival date rather than departure date decides. Removes December 2022 except the 24 arrivals that roll into 1 January, and the 49 June 2024 arrivals that roll past 1 July |

Of the in-window scheduled arrivals, 424,670 (97.23%) actually landed; 10,389 were
cancelled and 1,634 diverted. These aggregate onto the same 26,256-row grid as the taxi
table, filling 20,914 of its hours. JFK schedules nothing to land in 1,910 of its hours and
LaGuardia in 3,432 — the latter is the overnight curfew, at 6.3 hours a day.

**Model table** (`data/curated/model_table.parquet`, `shapes_model_table.json`)

| Quantity | Value |
|---|---|
| Rows × columns | 26,256 × 79 |
| Declared features | 43, across 10 groups in `model_table_roles.json` |
| Train / test split | 17,520 rows (2023) / 8,736 rows (Jan–Jun 2024) |
| Rows without a full week of lag history | 336 (the first week of 2023, flagged `has_full_lags`) |
| Airport-hours with no pickups | 1,180, where the value target `mean_total` is undefined |
| Precipitation not observed at the site | 1,527 hours (5.8%), resolved from a neighbouring station or assumed dry |
| Seasonal-naive benchmark, test period | 52.8 RMSE / 36.1 MAE pickups; $10.13 RMSE / $5.18 MAE on `mean_total` |

Every join in 2c asserts the row count, the leakage contract from
`flight_column_roles.json` is enforced programmatically, and a null audit requires every
null in the table to belong to a documented family before the file is written.

## Notes and known limitations

- **BTS covers domestic flights only.** JFK converts 193,143 scheduled domestic arrivals
  into 2,753,043 taxi pickups (14.3 per arrival) while LaGuardia converts 243,611 into
  1,900,138 (7.8 per arrival). The gap is largely JFK's international traffic, which is
  absent from the flight features entirely, and partly its long-haul mix. Discussed in the
  report.
- **The study window coincides with New York's record snow drought.** Freezing
  precipitation occurs in roughly 0.3% of airport-hours, so the weather signal here is
  rain-dominated and any snow effect is weakly identified.
- The two busiest scheduled hours differ by airport: JFK peaks at 14:00 and 07:00, LGA at
  13:00 and 17:00. LaGuardia's `share_longhaul` is 0.034 against JFK's 0.385, which is the
  perimeter rule appearing in the data without being told to.
- Mean arrival delay is 22.5 minutes at JFK against 17.5 at LaGuardia, and cancellation
  rates are 2.1% and 2.5% respectively.
- The value model is undefined in the 1,180 hours with no pickups, which are overwhelmingly
  overnight. It is therefore silent exactly where the demand model says not to go.
- Newark enters the study only as flight-side context: yellow taxis may drop off there but
  may not pick up, so it is never part of the response.