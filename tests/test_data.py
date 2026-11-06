from pathlib import Path

import pandas as pd

from download_data import EXPECTED_SHA256, sha256
from hotel_ml import (
    CATEGORICAL_FEATURES,
    FEATURES,
    LEAKAGE_OR_UNAVAILABLE,
    NUMERIC_FEATURES,
    TARGET,
    load_raw,
    prepare,
    time_split,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hotel_bookings.csv"


def test_dataset_checksum_and_schema():
    assert DATA.exists(), "Run python download_data.py first."
    assert sha256(DATA) == EXPECTED_SHA256
    frame = load_raw(DATA)
    assert len(frame) == 119390
    assert TARGET in frame.columns
    for column in FEATURES + LEAKAGE_OR_UNAVAILABLE:
        assert column in frame.columns


def test_model_features_exclude_leakage_columns():
    assert set(FEATURES).isdisjoint(LEAKAGE_OR_UNAVAILABLE)
    assert "reservation_status" not in FEATURES
    assert "reservation_status_date" not in FEATURES
    assert "assigned_room_type" not in FEATURES
    assert TARGET not in FEATURES


def test_time_split_is_ordered_and_disjoint():
    prepared, _notes = prepare(load_raw(DATA))
    prepared = prepared.copy()
    prepared["row_id"] = pd.RangeIndex(len(prepared))
    train, val, test = time_split(prepared)
    groups = [set(train["row_id"]), set(val["row_id"]), set(test["row_id"])]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert len(train) + len(val) + len(test) == len(prepared)
    assert train["booking_date"].max() <= val["booking_date"].min()
    assert val["booking_date"].max() <= test["booking_date"].min()
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(FEATURES)
