# NOAA Weather Dataset Source Notes

Use this file to track exact external weather provenance for reproducibility.

## Record format
- Source URL:
- Access date:
- File name:
- Time range covered:
- Station used:
- Relevant columns used:
- Known limitations:

## Current entry
- Source URL: `https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00014734&startDate=2025-01-01&endDate=2026-02-28&dataTypes=TAVG,TMAX,TMIN,PRCP,SNOW,AWND&includeAttributes=false&units=metric&format=json`
- Access date: `2026-03-26`
- File name: `noaa_daily_usw00014734_20250101_20260228.csv`
- Time range covered: `2025-01-01` through `2026-02-28`
- Station used: NOAA station `USW00014734`, verified in the official ISD station-history file as `NEWARK LIBERTY INTERNATIONAL AP` (`KEWR`)
- Relevant columns used: `DATE`, `TAVG`, `TMAX`, `TMIN`, `PRCP`, `SNOW`, `AWND`
- Known limitations:
  - The current RL environment uses daily weather summaries, not hourly weather observations, so intraday weather shocks are not represented directly.
  - `TAVG` and `AWND` have a small number of missing values in the downloaded range; the feature loader currently fills missing temperature and wind values with in-range medians and fills missing precipitation/snow with `0.0`.

## Station verification source
- Source URL: `https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv`
- Access date: `2026-03-26`
- Verified row: `"725020","14734","NEWARK LIBERTY INTERNATIONAL AP","US","NJ","KEWR","+40.683","-074.169","+0002.0","19730101","20250825"`
