"""Streaming structural and value checks for the three M5 source files."""
from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
import tempfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from . import config as C


def _logger() -> logging.Logger:
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m5.validation")
    if not logger.handlers:
        handler = logging.FileHandler(C.VALIDATION_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _read_header(path: Path, required: set[str]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required M5 file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), None)
    if not header:
        raise ValueError(f"CSV is empty or has no header: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"{path.name} contains duplicate column names")
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return header


def _records(path: Path, header: list[str]):
    """Yield row dictionaries while rejecting ragged CSV records."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        next(reader, None)
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise ValueError(
                    f"{path.name} has a malformed record at line {line_number}: "
                    f"expected {len(header)} fields, found {len(values)}"
                )
            yield dict(zip(header, values))


def validate_m5_data() -> dict:
    """Validate M5 files using streaming CSV reads and bounded auxiliary state.

    Sales are never reshaped to long format. Price-key uniqueness is checked
    with a temporary disk-backed SQLite primary key so memory does not grow
    with the 6.8 million price rows.
    """
    log = _logger()
    log.info("Starting streaming M5 source validation")
    sales_days = [f"d_{i}" for i in range(1, C.LAST_SALES_DAY + 1)]
    sales_header = _read_header(C.SALES_PATH, set(C.SERIES_ID_COLUMNS) | set(sales_days))
    calendar_header = _read_header(C.CALENDAR_PATH, {"date", "wm_yr_wk", "d"})
    price_header = _read_header(C.PRICES_PATH, {"store_id", "item_id", "wm_yr_wk", "sell_price"})
    if sales_header[-len(sales_days):] != sales_days:
        raise ValueError("Sales day columns must appear in order from d_1 through d_1941 at the end of the file")

    errors: list[str] = []
    sales_rows = duplicate_series_ids = 0
    missing_sales = invalid_sales = empty_sales_ids = 0
    seen_series: set[str] = set()
    item_ids: set[str] = set()
    store_ids: set[str] = set()
    sales_pairs: set[tuple[str, str]] = set()
    min_sales: int | None = None
    max_sales: int | None = None
    for row in _records(C.SALES_PATH, sales_header):
        sales_rows += 1
        identifiers = [row[col].strip() for col in C.SERIES_ID_COLUMNS]
        if any(not value for value in identifiers):
            empty_sales_ids += 1
        series_id, item_id, _, _, store_id, _ = identifiers
        if series_id in seen_series:
            duplicate_series_ids += 1
        seen_series.add(series_id)
        item_ids.add(item_id)
        store_ids.add(store_id)
        sales_pairs.add((store_id, item_id))
        for col in sales_days:
            value = row[col].strip()
            if not value:
                missing_sales += 1
                continue
            try:
                number = int(value)
                if number < 0:
                    invalid_sales += 1
                min_sales = number if min_sales is None else min(min_sales, number)
                max_sales = number if max_sales is None else max(max_sales, number)
            except ValueError:
                invalid_sales += 1

    calendar_rows = 0
    day_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    week_ids: set[str] = set()
    ordered_calendar: list[tuple[str, date]] = []
    invalid_calendar_dates = empty_calendar_ids = 0
    for row in _records(C.CALENDAR_PATH, calendar_header):
        calendar_rows += 1
        day_id, date_text, week_id = row["d"].strip(), row["date"].strip(), row["wm_yr_wk"].strip()
        if not day_id or not week_id:
            empty_calendar_ids += 1
        day_counts[day_id] += 1
        date_counts[date_text] += 1
        if week_id:
            week_ids.add(week_id)
        try:
            parsed = date.fromisoformat(date_text)
            ordered_calendar.append((day_id, parsed))
        except ValueError:
            invalid_calendar_dates += 1

    missing_sales_day_ids = [day for day in sales_days if day_counts[day] == 0]
    multiply_mapped_sales_day_ids = [day for day in sales_days if day_counts[day] > 1]
    duplicate_day_ids = sorted(day for day, count in day_counts.items() if count > 1)
    duplicate_dates = sorted(value for value, count in date_counts.items() if count > 1)
    calendar_order_errors = 0
    calendar_gap_errors = 0
    day_sequence_errors = 0
    for previous, current in zip(ordered_calendar, ordered_calendar[1:]):
        if current[1] <= previous[1]:
            calendar_order_errors += 1
        if current[1] != previous[1] + timedelta(days=1):
            calendar_gap_errors += 1
        try:
            if int(current[0].removeprefix("d_")) != int(previous[0].removeprefix("d_")) + 1:
                day_sequence_errors += 1
        except ValueError:
            day_sequence_errors += 1
    if any(day_counts[d] != 1 for d in sales_days):
        errors.append("Every sales day ID must map to exactly one calendar record")
    if duplicate_day_ids or duplicate_dates or invalid_calendar_dates or empty_calendar_ids:
        errors.append("Calendar day IDs and dates must be nonempty, unique, and valid")
    if calendar_order_errors or calendar_gap_errors or day_sequence_errors:
        errors.append("Calendar rows and day IDs must be chronological and continuous")

    price_rows = invalid_prices = empty_price_keys = unknown_price_weeks = 0
    unknown_price_ids = 0
    duplicate_price_keys = 0
    priced_pairs: set[tuple[str, str]] = set()
    # SQLite uses disk storage in this temporary directory, keeping duplicate
    # detection bounded in RAM. The temporary database is deleted on exit.
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m5_price_keys_", dir=C.OUTPUTS_DIR) as temp_dir:
        db = sqlite3.connect(str(Path(temp_dir) / "keys.sqlite"))
        db.execute("CREATE TABLE price_keys (store_id TEXT, item_id TEXT, wm_yr_wk TEXT, PRIMARY KEY(store_id,item_id,wm_yr_wk)) WITHOUT ROWID")
        pending = []
        for row in _records(C.PRICES_PATH, price_header):
            price_rows += 1
            store_id, item_id, week_id = (row[k].strip() for k in ("store_id", "item_id", "wm_yr_wk"))
            if not store_id or not item_id or not week_id:
                empty_price_keys += 1
                continue
            if week_id not in week_ids:
                unknown_price_weeks += 1
            if item_id not in item_ids or store_id not in store_ids:
                unknown_price_ids += 1
            else:
                priced_pairs.add((store_id, item_id))
            try:
                price = float(row["sell_price"])
                if not math.isfinite(price) or price <= 0:
                    invalid_prices += 1
            except (TypeError, ValueError):
                invalid_prices += 1
            pending.append((store_id, item_id, week_id))
            if len(pending) >= 10_000:
                duplicate_price_keys += _insert_price_keys(db, pending)
                pending.clear()
        if pending:
            duplicate_price_keys += _insert_price_keys(db, pending)
        db.close()

    missing_price_pairs = len(sales_pairs - priced_pairs)
    if (duplicate_series_ids or empty_sales_ids or missing_sales or invalid_sales or not sales_rows):
        errors.append("Sales records must have unique IDs, nonempty identifiers, and valid nonnegative daily values")
    if duplicate_price_keys or empty_price_keys or unknown_price_weeks or unknown_price_ids or invalid_prices or missing_price_pairs:
        errors.append("Price records must have unique valid keys, positive finite prices, known weeks, and cover every sales pair")

    report = {
        "status": "failed" if errors else "passed",
        "files": {
            "sales": {"path": str(C.SALES_PATH.relative_to(C.ROOT_DIR)), "rows": sales_rows,
                      "columns": len(sales_header), "daily_columns": len(sales_days),
                      "duplicate_series_ids": duplicate_series_ids, "empty_identifier_rows": empty_sales_ids,
                      "missing_sales_cells": missing_sales, "invalid_sales_cells": invalid_sales,
                      "sales_min": min_sales, "sales_max": max_sales,
                      "unique_store_item_pairs": len(sales_pairs)},
            "calendar": {"path": str(C.CALENDAR_PATH.relative_to(C.ROOT_DIR)), "rows": calendar_rows,
                         "unique_day_ids": len(day_counts), "unique_dates": len(date_counts),
                         "invalid_dates": invalid_calendar_dates, "empty_day_or_week_ids": empty_calendar_ids,
                         "duplicate_day_ids": len(duplicate_day_ids), "duplicate_dates": len(duplicate_dates),
                         "missing_sales_day_ids": missing_sales_day_ids,
                         "multiply_mapped_sales_day_ids": multiply_mapped_sales_day_ids,
                         "chronological_order_errors": calendar_order_errors,
                         "date_continuity_errors": calendar_gap_errors,
                         "day_sequence_errors": day_sequence_errors},
            "prices": {"path": str(C.PRICES_PATH.relative_to(C.ROOT_DIR)), "rows": price_rows,
                       "invalid_or_nonpositive_prices": invalid_prices, "empty_key_rows": empty_price_keys,
                       "duplicate_price_keys": duplicate_price_keys, "unknown_calendar_weeks": unknown_price_weeks,
                       "unknown_item_or_store_rows": unknown_price_ids,
                       "sales_pairs_without_any_price": missing_price_pairs},
        },
        "scope_note": "Streaming validation; sales remain wide. Duplicate price keys use a temporary disk-backed index.",
    }
    C.VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Validation completed: %s (%d series, %d price rows, %d errors)",
             report["status"], sales_rows, price_rows, len(errors))
    if errors:
        report["errors"] = errors
        C.VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise ValueError("M5 validation failed: " + "; ".join(errors))
    return report


def _insert_price_keys(db: sqlite3.Connection, keys: list[tuple[str, str, str]]) -> int:
    duplicates = 0
    for key in keys:
        cursor = db.execute("INSERT OR IGNORE INTO price_keys VALUES (?, ?, ?)", key)
        duplicates += cursor.rowcount == 0
    db.commit()
    return duplicates
