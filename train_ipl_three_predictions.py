
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "ipl_ball_by_ball_standardized.csv"
MODEL_PATH = PROJECT_DIR / "models" / "ipl_three_predictions.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "ipl_three_predictions_metrics.json"

# Ball 48 means 8.0 overs completed; ball 66 means 11.0; ball 102 means 17.0.
FIRST_INNINGS_SCORE_START_BALL = 66
LIVE_MODEL_START_BALL = 48
FINAL_THREE_OVERS_START_BALL = 102

CATEGORICAL = ["batting_team", "bowling_team", "venue"]
SCORE_NUMERIC = [
    "legal_ball_number",
    "innings_runs_so_far",
    "innings_wickets_so_far",
    "current_run_rate",
]
CHASE_NUMERIC = [
    "legal_ball_number",
    "balls_remaining",
    "target",
    "innings_runs_so_far",
    "innings_wickets_so_far",
    "runs_required",
    "current_run_rate",
    "required_run_rate",
]


def chronological_masks(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return early-training, calibration, and latest holdout masks by match date."""
    matches = data[["match_id", "date"]].drop_duplicates().sort_values(["date", "match_id"])
    early_ids = set(matches.iloc[: int(len(matches) * 0.64)]["match_id"])
    train_ids = set(matches.iloc[: int(len(matches) * 0.80)]["match_id"])
    early_mask = data["match_id"].isin(early_ids)
    train_mask = data["match_id"].isin(train_ids)
    calibration_mask = train_mask & ~early_mask
    return early_mask, calibration_mask, ~train_mask


def classifier_pipeline(numeric: list[str]) -> Pipeline:
    """Build an XGBoost classifier using only live, user-entered match state."""
    preprocess = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numbers", "passthrough", numeric),
        ]
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=650,
        max_depth=5,
        learning_rate=0.035,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def score_regressor_pipeline() -> Pipeline:
    """Build an XGBoost regression model for final first-innings runs."""
    preprocess = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numbers", "passthrough", SCORE_NUMERIC),
        ]
    )
    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        max_depth=5,
        learning_rate=0.035,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def first_innings_live_rows(data: pd.DataFrame, start_ball: int) -> pd.DataFrame:
    """Label every live first-innings state with that innings' final score."""
    first = data[(data["innings"].eq(1)) & (data["legal_ball"].eq(1))].copy()
    first = first.dropna(subset=CATEGORICAL + SCORE_NUMERIC)
    first = first.sort_values(["match_id", "legal_ball_number", "over", "ball_in_over"])
    first["final_first_innings_score"] = first.groupby("match_id")["innings_runs_so_far"].transform("last")
    final_ball = first.groupby("match_id")["legal_ball_number"].transform("max")
    return first[
        first["legal_ball_number"].lt(final_ball)
        & first["legal_ball_number"].ge(start_ball)
    ].copy()


def completed_chase_rows(data: pd.DataFrame, start_ball: int) -> pd.DataFrame:
    """Return valid live-chase states from a stated point in the innings onward."""
    chase = data[
        data["innings"].eq(2)
        & data["legal_ball"].eq(1)
        & data["legal_ball_number"].ge(start_ball)
    ].copy()
    chase = chase.dropna(subset=CATEGORICAL + CHASE_NUMERIC + ["batting_team_won", "match_winner"])
    valid_result = chase["match_winner"].eq(chase["team1"]) | chase["match_winner"].eq(chase["team2"])
    return chase.loc[valid_result].copy()


def choose_threshold(probabilities: np.ndarray, outcomes: pd.Series) -> float:
    """Select the classification cutoff from calibration matches only."""
    choices = np.arange(0.30, 0.701, 0.01)
    return float(max(choices, key=lambda cutoff: accuracy_score(outcomes, probabilities >= cutoff)))


def train_chase_classifier(data: pd.DataFrame, model_name: str) -> tuple[Pipeline, dict[str, Any]]:
    """Calibrate on past matches, test on future matches, then train final model."""
    features = CATEGORICAL + CHASE_NUMERIC
    outcomes = data["batting_team_won"].astype(int)
    early_mask, calibration_mask, test_mask = chronological_masks(data)

    calibration_model = classifier_pipeline(CHASE_NUMERIC)
    calibration_model.fit(data.loc[early_mask, features], outcomes.loc[early_mask])
    calibration_probability = calibration_model.predict_proba(data.loc[calibration_mask, features])[:, 1]
    threshold = choose_threshold(calibration_probability, outcomes.loc[calibration_mask])

    validation_train_mask = ~test_mask
    validation_model = classifier_pipeline(CHASE_NUMERIC)
    validation_model.fit(data.loc[validation_train_mask, features], outcomes.loc[validation_train_mask])
    probabilities = validation_model.predict_proba(data.loc[test_mask, features])[:, 1]
    validation_outcomes = outcomes.loc[test_mask]
    metrics: dict[str, Any] = {
        "model": model_name,
        "training_matches": int(data.loc[validation_train_mask, "match_id"].nunique()),
        "validation_matches": int(data.loc[test_mask, "match_id"].nunique()),
        "training_rows": int(validation_train_mask.sum()),
        "validation_rows": int(test_mask.sum()),
        "decision_threshold": round(threshold, 2),
        "chronological_holdout_accuracy": round(
            float(accuracy_score(validation_outcomes, probabilities >= threshold)), 4
        ),
        "chronological_holdout_accuracy_at_0_50": round(
            float(accuracy_score(validation_outcomes, probabilities >= 0.50)), 4
        ),
        "chronological_holdout_roc_auc": round(float(roc_auc_score(validation_outcomes, probabilities)), 4),
    }
    final_model = classifier_pipeline(CHASE_NUMERIC)
    final_model.fit(data[features], outcomes)
    return final_model, metrics


def train_score_regressor(data: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    """Train a first-innings run regressor and evaluate on the latest matches."""
    features = CATEGORICAL + SCORE_NUMERIC
    _, _, test_mask = chronological_masks(data)
    train_mask = ~test_mask
    validation_model = score_regressor_pipeline()
    validation_model.fit(data.loc[train_mask, features], data.loc[train_mask, "final_first_innings_score"])
    prediction = validation_model.predict(data.loc[test_mask, features])
    actual = data.loc[test_mask, "final_first_innings_score"].to_numpy()
    absolute_error = np.abs(actual - prediction)
    metrics: dict[str, Any] = {
        "training_matches": int(data.loc[train_mask, "match_id"].nunique()),
        "validation_matches": int(data.loc[test_mask, "match_id"].nunique()),
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(test_mask.sum()),
        "chronological_holdout_mae_runs": round(float(mean_absolute_error(actual, prediction)), 2),
        "chronological_holdout_rmse_runs": round(float(mean_squared_error(actual, prediction) ** 0.5), 2),
        "within_10_runs": round(float((absolute_error <= 10).mean()), 4),
        "within_15_runs": round(float((absolute_error <= 15).mean()), 4),
        "within_20_runs": round(float((absolute_error <= 20).mean()), 4),
    }
    final_model = score_regressor_pipeline()
    final_model.fit(data[features], data["final_first_innings_score"])
    return final_model, metrics


def overs_to_balls(overs: float) -> int:
    """Convert cricket notation such as 10.4 to a legal-ball count of 64."""
    complete_overs = int(overs)
    balls = int(round((float(overs) - complete_overs) * 10))
    if balls not in range(6):
        raise ValueError("Use cricket overs notation, for example 10.4 rather than 10.6.")
    return complete_overs * 6 + balls


def load_models(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the single saved model bundle for the Streamlit application."""
    return joblib.load(model_path)


def _prediction_frame(values: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: values[column] for column in columns}])


def predict_first_innings_score(
    models: dict[str, Any], *, batting_team: str, bowling_team: str, venue: str,
    current_runs: int, wickets: int, overs: float,
) -> dict[str, int]:
    """Return a final-score estimate and validation-based plausible range."""
    legal_balls = overs_to_balls(overs)
    start_ball = int(models["model_starts"]["first_innings_score"])
    if legal_balls < start_ball:
        raise ValueError(f"This score model begins after {start_ball // 6}.0 overs of the first innings.")
    if legal_balls > 119:
        raise ValueError("Enter a first-innings state through 19.5 overs.")
    current_rate = current_runs * 6 / legal_balls
    row = _prediction_frame(
        {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "legal_ball_number": legal_balls,
            "innings_runs_so_far": current_runs,
            "innings_wickets_so_far": wickets,
            "current_run_rate": current_rate,
        },
        CATEGORICAL + SCORE_NUMERIC,
    )
    projection = max(float(current_runs), float(models["first_innings_score_model"].predict(row)[0]))
    margin = float(models["metrics"]["first_innings_score"]["chronological_holdout_mae_runs"])
    return {
        "projected_final_score": round(projection),
        "likely_low": max(current_runs, round(projection - margin)),
        "likely_high": round(projection + margin),
    }


def _predict_chase(
    models: dict[str, Any], *, model_key: str, metrics_key: str, chasing_team: str,
    defending_team: str, venue: str, target: int, current_runs: int, wickets: int, overs: float,
) -> dict[str, float | str]:
    legal_balls = overs_to_balls(overs)
    start_ball = int(models["model_starts"][metrics_key])
    if legal_balls < start_ball:
        required_overs = f"{start_ball // 6}.0"
        raise ValueError(f"This model begins after {required_overs} overs of the chase.")
    if legal_balls > 119:
        raise ValueError("Enter a live chase state through 19.5 overs.")
    balls_remaining = 120 - legal_balls
    runs_required = max(target - current_runs, 0)
    current_rate = current_runs * 6 / legal_balls
    required_rate = runs_required * 6 / max(balls_remaining, 1)
    row = _prediction_frame(
        {
            "batting_team": chasing_team,
            "bowling_team": defending_team,
            "venue": venue,
            "legal_ball_number": legal_balls,
            "balls_remaining": balls_remaining,
            "target": target,
            "innings_runs_so_far": current_runs,
            "innings_wickets_so_far": wickets,
            "runs_required": runs_required,
            "current_run_rate": current_rate,
            "required_run_rate": required_rate,
        },
        CATEGORICAL + CHASE_NUMERIC,
    )
    probability = float(models[model_key].predict_proba(row)[0, 1])
    return {
        "chasing_team": chasing_team,
        "chasing_team_win_probability": round(probability * 100, 1),
        "defending_team": defending_team,
        "defending_team_win_probability": round((1 - probability) * 100, 1),
    }


def predict_live_match_winner(
    models: dict[str, Any], *, chasing_team: str, defending_team: str, venue: str,
    target: int, current_runs: int, wickets: int, overs: float,
) -> dict[str, float | str]:
    """Return second-innings probabilities from eight completed overs onward."""
    return _predict_chase(
        models,
        model_key="live_second_innings_model",
        metrics_key="live_second_innings_after_8_overs",
        chasing_team=chasing_team,
        defending_team=defending_team,
        venue=venue,
        target=target,
        current_runs=current_runs,
        wickets=wickets,
        overs=overs,
    )


def predict_final_three_overs(
    models: dict[str, Any], *, chasing_team: str, defending_team: str, venue: str,
    target: int, current_runs: int, wickets: int, overs: float,
) -> dict[str, float | str]:
    """Return final-three-overs probabilities from 17.0 overs onward."""
    return _predict_chase(
        models,
        model_key="final_three_overs_model",
        metrics_key="final_three_overs",
        chasing_team=chasing_team,
        defending_team=defending_team,
        venue=venue,
        target=target,
        current_runs=current_runs,
        wickets=wickets,
        overs=overs,
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {DATA_PATH}")
    raw_data = pd.read_csv(DATA_PATH, low_memory=False)
    raw_data["date"] = pd.to_datetime(raw_data["date"], errors="coerce")
    raw_data = raw_data.dropna(subset=["match_id", "date"]).copy()

    score_model, score_metrics = train_score_regressor(
        first_innings_live_rows(raw_data, FIRST_INNINGS_SCORE_START_BALL)
    )
    live_model, live_metrics = train_chase_classifier(
        completed_chase_rows(raw_data, LIVE_MODEL_START_BALL), "live_second_innings_after_8_overs"
    )
    final_model, final_metrics = train_chase_classifier(
        completed_chase_rows(raw_data, FINAL_THREE_OVERS_START_BALL), "final_three_overs"
    )

    # The user accepts a score estimate within ±20 runs, so this operational
    # summary expresses that model with the same pass/fail idea as the two
    # classification models. It is not a standard regression accuracy metric.
    operational_average = float(
        np.mean(
            [
                score_metrics["within_20_runs"],
                live_metrics["chronological_holdout_accuracy"],
                final_metrics["chronological_holdout_accuracy"],
            ]
        )
    )
    metrics: dict[str, Any] = {
        "data": {"source": str(DATA_PATH), "matches": int(raw_data["match_id"].nunique())},
        "first_innings_score": {
            **score_metrics,
            "starts_after_overs": "11.0",
        },
        "live_second_innings_after_8_overs": live_metrics,
        "final_three_overs": final_metrics,
        "operational_three_model_average": {
            "score_tolerance": "first-innings score is considered correct when within ±20 runs",
            "average": round(operational_average, 4),
            "warning": "This is an operational summary, not a standard single-model accuracy metric.",
        },
    }

    model_bundle = {
        "first_innings_score_model": score_model,
        "live_second_innings_model": live_model,
        "final_three_overs_model": final_model,
        "model_starts": {
            "first_innings_score": FIRST_INNINGS_SCORE_START_BALL,
            "live_second_innings_after_8_overs": LIVE_MODEL_START_BALL,
            "final_three_overs": FINAL_THREE_OVERS_START_BALL,
        },
        "metrics": metrics,
        "feature_sets": {
            "first_innings_score": CATEGORICAL + SCORE_NUMERIC,
            "live_second_innings_after_8_overs": CATEGORICAL + CHASE_NUMERIC,
            "final_three_overs": CATEGORICAL + CHASE_NUMERIC,
        },
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model bundle: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"First-innings score MAE: {score_metrics['chronological_holdout_mae_runs']:.2f} runs")
    print(f"Second-innings  accuracy: {live_metrics['chronological_holdout_accuracy']:.2%}")
    print(f"Final-three-overs accuracy: {final_metrics['chronological_holdout_accuracy']:.2%}")
    print(f" average accuracy: {operational_average:.2%}")
if __name__ == "__main__":
    main()
