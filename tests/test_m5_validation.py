"""Focused tests for isolated M5 source validation."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from m5_pipeline import config as C
from m5_pipeline.validation import validate_m5_data


@pytest.fixture
def m5_files(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    sales_path = data / "sales.csv"
    calendar_path = data / "calendar.csv"
    prices_path = data / "prices.csv"
    sales_header = [*C.SERIES_ID_COLUMNS, "d_1", "d_2", "d_3"]
    calendar_header = ["date", "wm_yr_wk", "d"]
    price_header = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    sales = [["item_store_1", "item_1", "dept_1", "cat_1", "store_1", "state_1", "0", "2", "1"]]
    calendar = [
        ["2011-01-29", "201104", "d_1"],
        ["2011-01-30", "201104", "d_2"],
        ["2011-01-31", "201105", "d_3"],
    ]
    prices = [["store_1", "item_1", "201104", "2.50"], ["store_1", "item_1", "201105", "2.50"]]

    def write(path, header, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def save():
        write(sales_path, sales_header, sales)
        write(calendar_path, calendar_header, calendar)
        write(prices_path, price_header, prices)

    monkeypatch.setattr(C, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(C, "SALES_PATH", sales_path)
    monkeypatch.setattr(C, "CALENDAR_PATH", calendar_path)
    monkeypatch.setattr(C, "PRICES_PATH", prices_path)
    monkeypatch.setattr(C, "LAST_SALES_DAY", 3)
    monkeypatch.setattr(C, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(C, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(C, "VALIDATION_LOG_PATH", tmp_path / "logs" / "validation.log")
    monkeypatch.setattr(C, "VALIDATION_REPORT_PATH", tmp_path / "outputs" / "validation.json")
    save()
    return {"sales": sales, "calendar": calendar, "prices": prices, "save": save,
            "sales_path": sales_path, "calendar_path": calendar_path, "prices_path": prices_path}


def test_valid_small_dataset_passes_and_reports(m5_files):
    report = validate_m5_data()
    assert report["status"] == "passed"
    assert report["files"]["sales"]["rows"] == 1
    assert report["files"]["calendar"]["missing_sales_day_ids"] == []
    assert report["files"]["prices"]["duplicate_price_keys"] == 0


def test_rejects_missing_or_ambiguous_sales_day_mapping(m5_files):
    m5_files["calendar"].append(["2011-02-01", "201105", "d_3"])
    m5_files["save"]()
    with pytest.raises(ValueError, match="exactly one calendar record"):
        validate_m5_data()


@pytest.mark.parametrize("replacement", [
    ["2011-01-29", "201104", "d_1"],  # duplicate date and day ID
    ["2011-02-01", "201104", "d_4"],  # date gap
    ["2011-01-28", "201104", "d_2"],  # out of chronological order
])
def test_rejects_duplicate_or_discontinuous_calendar(m5_files, replacement):
    m5_files["calendar"][1] = replacement
    m5_files["save"]()
    with pytest.raises(ValueError, match="Calendar"):
        validate_m5_data()


def test_rejects_price_pair_not_present_in_sales(m5_files):
    m5_files["prices"][0][0] = "another_store"
    m5_files["save"]()
    with pytest.raises(ValueError, match="Price records"):
        validate_m5_data()


def test_rejects_price_week_not_in_calendar(m5_files):
    m5_files["prices"][0][2] = "209999"
    m5_files["save"]()
    with pytest.raises(ValueError, match="Price records"):
        validate_m5_data()


@pytest.mark.parametrize("bad_price", ["NaN", "inf", "0", "-1"])
def test_rejects_nonfinite_or_nonpositive_prices(m5_files, bad_price):
    m5_files["prices"][0][3] = bad_price
    m5_files["save"]()
    with pytest.raises(ValueError, match="Price records"):
        validate_m5_data()


def test_rejects_duplicate_price_key(m5_files):
    m5_files["prices"].append(m5_files["prices"][0].copy())
    m5_files["save"]()
    with pytest.raises(ValueError, match="Price records"):
        validate_m5_data()


def test_rejects_empty_sales_identifier(m5_files):
    m5_files["sales"][0][1] = ""
    m5_files["save"]()
    with pytest.raises(ValueError, match="Sales records"):
        validate_m5_data()


def test_rejects_ragged_sales_record(m5_files):
    m5_files["sales_path"].write_text(
        "id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2,d_3\nitem_store_1,item_1,dept_1,cat_1,store_1,state_1,0,2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed record"):
        validate_m5_data()
