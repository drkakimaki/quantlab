# Economic calendar dataset (USD important events)

This folder is intended to hold a machine-readable economic calendar for **high-impact USD events** (e.g. CPI, NFP, FOMC, GDP, Retail Sales, Core PCE, ISM, etc.).

## Files

- `usd_important_events.csv` — main dataset (one row per release/event)
- `usd_important_events.schema.json` — informal schema
- `sources.md` — where the dates come from (links + notes)

## CSV columns (proposed)

- `ts_utc` (ISO8601, e.g. `2026-03-12T12:30:00Z`) — event timestamp in UTC
- `country` (e.g. `US`)
- `ccy` (e.g. `USD`)
- `event` (short name, e.g. `CPI YoY`, `Nonfarm Payrolls`, `FOMC Rate Decision`)
- `category` (e.g. `inflation`, `labor`, `rates`, `growth`)
- `importance` (`high` | `medium` | `low`) — we will keep only `high` for this dataset
- `source` (string)
- `notes` (string)

## How to populate

This repo currently has **no working web search/browser tooling**, so we cannot automatically download calendars from the internet.

Two options:
1) Provide a source export (CSV/ICS) from your preferred calendar provider (ForexFactory/Investing/TradingEconomics/etc.).
2) Enable web_search (Brave API key) or a browser relay, then we can fetch official schedules (BLS/BEA/FOMC) and build the dataset.
