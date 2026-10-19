"""Train the full-data pre-match IPL XGBoost pipeline used by app.py.

Inputs are available after the toss: team 1, team 2, venue, toss winner and
toss decision. Result columns are excluded to prevent target leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "clean_ipl_matches.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "ipl_pre_match_xgboost.joblib"
METADATA_PATH = MODEL_DIR / "ipl_pre_match_model_metadata.json"
FEATURES = ["team1", "team2", "venue", "toss_winner", "toss_decision"]


def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return completed matches and a 1/0 target for a team-1 victory."""
    frame = data.copy()
    for column in ["team1", "team2", "toss_winner", "winner"]:
        frame[column] = frame[column].astype("string").str.strip().replace(
            {"Royal Challengers Bangalore": "Royal Challengers Bengaluru"}
        )
    frame["venue"] = frame["venue"].astype("string").str.strip()
    frame["toss_decision"] = frame["toss_decision"].astype("string").str.strip()

    # A missing winner is a no-result/abandoned match, not a team-1 loss.
    frame = frame.dropna(subset=FEATURES + ["winner"])
    frame = frame[frame["winner"].isin(set(frame["team1"]) | set(frame["team2"]))].copy()
    target = (frame["winner"] == frame["team1"]).astype(int)
    return frame[FEATURES], target


def build_model() -> Pipeline:
    encoder = ColumnTransformer(
        [("match_context", OneHotEncoder(handle_unknown="ignore"), FEATURES)],
        remainder="drop",
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", encoder), ("model", classifier)])


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    X, y = prepare_training_data(raw)

    # Evaluate against the newest 20% of matches so the score reflects a
    # realistic future-style prediction task instead of a random row split.
    dates = pd.to_datetime(raw.loc[X.index, "date"], errors="coerce")
    ordered_index = dates.sort_values().index
    X_ordered = X.loc[ordered_index]
    y_ordered = y.loc[ordered_index]
    split_at = int(len(X_ordered) * 0.80)
    validation_model = build_model()
    validation_model.fit(X_ordered.iloc[:split_at], y_ordered.iloc[:split_at])
    validation_accuracy = accuracy_score(
        y_ordered.iloc[split_at:], validation_model.predict(X_ordered.iloc[split_at:])
    )

    # Refit on every completed match before saving the model used by the app.
    pipeline = build_model()
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model": "XGBoost binary classifier",
                "target": "winner == team1",
                "training_rows": int(len(X)),
                "validation_rows": int(len(X_ordered) - split_at),
                "chronological_holdout_accuracy": round(float(validation_accuracy), 4),
                "features": FEATURES,
                "excluded_result_columns": ["winner", "player_of_match", "team1_won", "toss_winner_won"],
                "note": "Fitted on all completed matches in clean_ipl_matches.csv.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved full-data model to {MODEL_PATH}")
    print(f"Fitted on {len(X):,} completed matches")
    print(f"Chronological holdout accuracy: {validation_accuracy:.2%}")


if __name__ == "__main__":
    main()
