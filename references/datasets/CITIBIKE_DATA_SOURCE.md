# Citi Bike Dataset Source Notes

Use this file to track exact dataset provenance for reproducibility.

## Record format
- Source URL:
- Access date:
- File name:
- Time range covered:
- Region/system covered:
- Relevant columns used:
- Known limitations:

## Example entry (fill when data is downloaded)
- Source URL: `https://tripdata.s3.amazonaws.com/JC-202602-citibike-tripdata.csv.zip` (official Citi Bike system-data download)
- Access date: `2026-03-26`
- File name: `JC-202602-citibike-tripdata.csv`
- Time range covered: observed `started_at` values range from `2026-01-31 20:30:15.084` through `2026-02-28 23:53:29.516`
- Region/system covered: Jersey City monthly extract (`JC`)
- Relevant columns used: `started_at`, `ended_at`, start/end station IDs, start/end station names
- Known limitations:
  - File naming indicates the February 2026 Jersey City extract, but the observed trip timestamps include rides that start near the month boundary.
  - Current preprocessing uses only timestamps and station IDs/names; other fields such as ride type, rider class, and lat/lng are not yet used.

- Source URL: `https://tripdata.s3.amazonaws.com/JC-202601-citibike-tripdata.zip` (official Citi Bike system-data download)
- Access date: `2026-03-26`
- File name: `JC-202601-citibike-tripdata.csv`
- Time range covered: observed `started_at` values range from `2025-12-31 11:40:30.940` through `2026-01-31 23:47:07.645`
- Region/system covered: Jersey City monthly extract (`JC`)
- Relevant columns used: `started_at`, `ended_at`, start/end station IDs, start/end station names
- Known limitations:
  - Jersey City monthly archives are not named consistently across months: January 2026 was published as `.zip`, while February 2026 used `.csv.zip`.
  - Current preprocessing uses only timestamps and station IDs/names; other fields such as ride type, rider class, and lat/lng are not yet used.
