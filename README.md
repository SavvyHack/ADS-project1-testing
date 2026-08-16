# NYC Yellow Taxi Airport Demand — MAST30034 Project 1

**Student:** [Tavish Balyan] | **Student ID:** [1615873]

Research goal.

- **Dataset:** NYC TLC Yellow Taxi trip records, January 2023 – June 2024
- **External dataset:** [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ),
  filtered to flights touching JFK, LGA, and EWR
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

Writes ~1 GB to `data/landing/` in roughly 10 minutes. The range above is
the default; results reported in `report/main.pdf` use exactly this range.
Downloads are idempotent — if the run is interrupted, re-run the same command
and completed files are skipped. `--help` lists all options.

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
US flights. Only flights touching JFK, LGA, or EWR are retained. The script prints the exact figure
per month.

## Running the pipeline

Execute in order. Each notebook reads the previous stage's output.

| Order | Notebook | Purpose | Reads | Writes |
|-------|----------|---------|-------|--------|
| 1 | `notebooks/1_preprocessing.ipynb` | Filtering, joins, feature construction | `data/landing/` | `data/raw/`, `data/curated/` |
| 2 | `notebooks/2_analysis.ipynb` | Distributions, outliers, geospatial visualisation | `data/curated/` | `plots/` |
| 3 | `notebooks/3_modelling.ipynb` | Poisson GLM and gradient-boosted trees | `data/curated/` | `models/`, `plots/` |

## Repository structure

```
├── data/                  # gitignored
│   ├── landing/           # untouched downloads
│   ├── raw/               # after filtering and type coercion
│   └── curated/           # model-ready, joined
├── notebooks/             # numbered, run in order
├── scripts/
│   └── download.py        # data acquisition
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

## Preprocessing summary

Record counts at each stage, mirroring the table in the report.

| Step | Rows | Removed | Justification |
|------|------|---------|---------------|
| Raw TLC ingest | | — | |
| Timestamp range filter | | | Records outside the stated window |
| Airport zone filter | | | `PULocationID` in {132, 138} |
| Invalid fare removal | | | See report §2 |
| Raw BTS ingest | | — | |
| NYC airport filter | | | `Origin` or `Dest` in {JFK, LGA, EWR} |
| Joined to flight data | | | |

## Notes and known limitations

- BTS covers domestic flights only; international arrivals at JFK are not
  represented. Discussed in the report.