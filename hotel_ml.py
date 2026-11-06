"""Leakage-safe training for hotel booking cancellation prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

LOGGER = logging.getLogger("hotel_ml")

SEED = 42
TARGET = "is_canceled"
LEAKAGE_OR_UNAVAILABLE = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "agent",
    "company",
]
NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python download_data.py"
        )
    frame = pd.read_csv(path)
    expected = set(FEATURES + LEAKAGE_OR_UNAVAILABLE + [TARGET])
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return frame


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean rows and add the booking date used only to order the split."""
    notes: dict = {"rows_raw": int(len(frame))}
    data = frame.copy()
    data = data.replace({"NULL": np.nan, "null": np.nan, "": np.nan})
    notes["exact_duplicate_rows"] = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    notes["rows_after_dedup"] = int(len(data))

    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["arrival_date"] = pd.to_datetime(
        data["arrival_date_year"].astype("Int64").astype(str)
        + "-"
        + data["arrival_date_month"].astype(str)
        + "-"
        + data["arrival_date_day_of_month"].astype("Int64").astype(str),
        format="%Y-%B-%d",
        errors="coerce",
    )
    invalid_dates = int(data["arrival_date"].isna().sum())
    notes["invalid_arrival_dates_removed"] = invalid_dates
    data = data.dropna(subset=["arrival_date", TARGET]).copy()

    guests = (
        data["adults"].fillna(0)
        + data["children"].fillna(0)
        + data["babies"].fillna(0)
    )
    notes["zero_guest_rows_removed"] = int((guests <= 0).sum())
    data = data.loc[guests > 0].copy()

    notes["negative_adr_set_to_missing"] = int((data["adr"] < 0).sum())
    data.loc[data["adr"] < 0, "adr"] = np.nan

    data["booking_date"] = data["arrival_date"] - pd.to_timedelta(
        data["lead_time"].fillna(0), unit="D"
    )
    data[TARGET] = data[TARGET].astype(int)
    notes["rows_model"] = int(len(data))
    notes["cancellation_rate"] = float(data[TARGET].mean())
    return data, notes


def time_split(
    data: pd.DataFrame, train_frac: float = 0.60, val_frac: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by estimated booking date. Later bookings stay in test."""
    ordered = data.sort_values(
        ["booking_date", "arrival_date"], kind="mergesort"
    ).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Time split produced an empty partition.")
    return train, val, test


def _linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=50,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=50,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _hgb_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _hgb(max_depth: int | None, learning_rate: float) -> HistGradientBoostingClassifier:
    cat_idx = list(range(len(NUMERIC_FEATURES), len(FEATURES)))
    return HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=120,
        random_state=SEED,
        early_stopping=False,
        categorical_features=cat_idx,
    )


def model_specs() -> list[tuple[str, Pipeline, dict]]:
    return [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("preprocess", _linear_preprocessor()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=400,
                            solver="liblinear",
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        (
            "decision_tree",
            Pipeline(
                steps=[
                    ("preprocess", _tree_preprocessor()),
                    (
                        "model",
                        DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            {"model__max_depth": [4, 8, 12]},
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("preprocess", _tree_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=60,
                            class_weight="balanced_subsample",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [8, 16],
                "model__min_samples_leaf": [5, 20],
            },
        ),
        (
            "hist_gradient_boosting",
            Pipeline(
                steps=[
                    ("preprocess", _hgb_preprocessor()),
                    ("model", _hgb(max_depth=6, learning_rate=0.1)),
                ]
            ),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 6],
            },
        ),
    ]


def _classification_scores(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.round(np.arange(0.20, 0.81, 0.05), 2):
        scores = _classification_scores(y_true, proba, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
            }
        )
        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            best_threshold = float(threshold)
    return best_threshold, rows


def run_experiment(
    data_path: Path,
    model_path: Path,
    reports_dir: Path,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = load_raw(data_path)
    prepared, clean_notes = prepare(raw)
    train, val, test = time_split(prepared)
    LOGGER.info(
        "Split sizes train=%s val=%s test=%s", len(train), len(val), len(test)
    )

    x_train = train[FEATURES]
    y_train = train[TARGET].to_numpy()
    x_val = val[FEATURES]
    y_val = val[TARGET].to_numpy()
    x_test = test[FEATURES]
    y_test = test[TARGET].to_numpy()

    splitter = TimeSeriesSplit(n_splits=3)
    comparison = []
    fitted = {}
    for name, pipeline, grid in model_specs():
        LOGGER.info("Tuning %s", name)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="roc_auc",
            cv=splitter,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)
        fitted[name] = search
        comparison.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_roc_auc_mean": float(search.best_score_),
                "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            }
        )
        LOGGER.info(
            "%s CV ROC AUC %.4f params %s",
            name,
            search.best_score_,
            search.best_params_,
        )

    comparison = sorted(comparison, key=lambda row: row["cv_roc_auc_mean"], reverse=True)
    winner_name = comparison[0]["model"]
    winner = fitted[winner_name].best_estimator_
    LOGGER.info("Selected %s by mean time-series CV ROC AUC", winner_name)

    train_proba = winner.predict_proba(x_train)[:, 1]
    val_proba = winner.predict_proba(x_val)[:, 1]
    threshold, threshold_rows = choose_threshold(y_val, val_proba)
    train_scores = _classification_scores(y_train, train_proba, threshold)
    val_scores = _classification_scores(y_val, val_proba, threshold)
    test_proba = winner.predict_proba(x_test)[:, 1]
    test_scores = _classification_scores(y_test, test_proba, threshold)

    reloaded_check = _roundtrip_check(winner, x_val.iloc[:25], model_path, threshold, winner_name)

    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    example = x_test.iloc[0].to_dict()
    example = {
        key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in example.items()
    }
    # JSON-friendly ints
    for key, value in list(example.items()):
        if isinstance(value, (np.integer,)):
            example[key] = int(value)
        elif isinstance(value, (np.floating,)):
            example[key] = float(value)

    summary = {
        "seed": SEED,
        "cleaning": clean_notes,
        "split": {
            "strategy": "chronological by estimated booking date (arrival date minus lead time)",
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_booking_date_min": str(train["booking_date"].min().date()),
            "train_booking_date_max": str(train["booking_date"].max().date()),
            "validation_booking_date_min": str(val["booking_date"].min().date()),
            "validation_booking_date_max": str(val["booking_date"].max().date()),
            "test_booking_date_min": str(test["booking_date"].min().date()),
            "test_booking_date_max": str(test["booking_date"].max().date()),
            "train_cancellation_rate": float(train[TARGET].mean()),
            "validation_cancellation_rate": float(val[TARGET].mean()),
            "test_cancellation_rate": float(test[TARGET].mean()),
        },
        "excluded_columns": LEAKAGE_OR_UNAVAILABLE,
        "features": FEATURES,
        "selection_metric": "mean ROC AUC from TimeSeriesSplit on the training period only",
        "operating_threshold_rule": "threshold on the validation period that maximizes F1 for cancellation",
        "model_comparison": comparison,
        "selected_model": winner_name,
        "selected_params": comparison[0]["best_params"],
        "threshold": threshold,
        "threshold_sweep_validation": threshold_rows,
        "train_at_selected_threshold": train_scores,
        "validation": val_scores,
        "test": test_scores,
        "serialization_check": reloaded_check,
        "example_request": example,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "example_request.json").write_text(
        json.dumps(example, indent=2), encoding="utf-8"
    )
    _save_importance(winner, x_val, y_val, reports_dir)
    _save_error_slices(val, val_proba, threshold, reports_dir / "validation_error_slices.csv")
    _save_error_slices(test, test_proba, threshold, reports_dir / "test_error_slices.csv")
    LOGGER.info(
        "Validation ROC AUC %.4f | Test ROC AUC %.4f | threshold %.2f",
        val_scores["roc_auc"],
        test_scores["roc_auc"],
        threshold,
    )
    return summary


def _roundtrip_check(model, sample: pd.DataFrame, model_path: Path, threshold: float, name: str) -> dict:
    before = model.predict_proba(sample)[:, 1]
    artifact = {"pipeline": model, "threshold": threshold, "features": FEATURES, "model_name": name}
    joblib.dump(artifact, model_path)
    loaded = joblib.load(model_path)
    after = loaded["pipeline"].predict_proba(sample[loaded["features"]])[:, 1]
    max_abs = float(np.max(np.abs(before - after)))
    if max_abs > 1e-8:
        raise RuntimeError(f"Reloaded model predictions differ by {max_abs}")
    return {"max_abs_probability_difference": max_abs, "rows_checked": int(len(sample))}


def _save_importance(model, x_val: pd.DataFrame, y_val: np.ndarray, reports_dir: Path) -> None:
    from sklearn.inspection import permutation_importance

    sample = x_val.iloc[:2000]
    target = y_val[: len(sample)]
    result = permutation_importance(
        model,
        sample,
        target,
        n_repeats=5,
        random_state=SEED,
        scoring="roc_auc",
        n_jobs=1,
    )
    table = (
        pd.DataFrame(
            {
                "feature": list(sample.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(reports_dir / "permutation_importance.csv", index=False)


def _save_error_slices(frame: pd.DataFrame, proba: np.ndarray, threshold: float, path: Path) -> None:
    pred = (proba >= threshold).astype(int)
    y = frame[TARGET].to_numpy()
    out = frame[["deposit_type", "hotel", "customer_type", "lead_time"]].copy()
    out["y_true"] = y
    out["y_pred"] = pred
    out["error_type"] = np.where(
        (out["y_true"] == 1) & (out["y_pred"] == 0),
        "false_negative",
        np.where((out["y_true"] == 0) & (out["y_pred"] == 1), "false_positive", "correct"),
    )
    summary = (
        out.groupby(["error_type", "deposit_type"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    summary.to_csv(path, index=False)
