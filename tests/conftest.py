"""Shared test setup: run from the project root with `python -m pytest`."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _isolated_logs(tmp_path_factory):
    """Keep test logs out of the project's logs/ folder."""
    C.LOG_PATH = tmp_path_factory.mktemp("logs") / "test.log"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite database."""
    monkeypatch.setattr(C, "DB_PATH", tmp_path / "test.db")


@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv(C.RAW_DATA_PATH)


@pytest.fixture(scope="session")
def demo_df():
    return pd.read_csv(C.DEMO_DATA_PATH)


def make_series(days=20, store="S001", product="P0001", start="2024-01-01", sales=None, inventory=200):
    """Small synthetic daily series for unit tests (sales default = 1, 2, 3, ...)."""
    dates = pd.date_range(start, periods=days, freq="D")
    sales = list(range(1, days + 1)) if sales is None else sales
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"), "Store ID": store, "Product ID": product,
        "Inventory Level": inventory, "Units Sold": sales, "Price": 10.0, "Discount": 5,
    })
