"""
NBA Game Predictor — Core ML Model
Uses an ensemble of XGBoost + Random Forest to predict:
  - Win probability for each team
  - Predicted final score for each team
  - Confidence interval on the score
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("scikit-learn / xgboost not installed.")

from config import (
    MODELS_DIR, FEATURE_COLUMNS, XGBOOST_PARAMS, RF_PARAMS, SCORE_MODEL_PARAMS
)

logger = logging.getLogger(__name__)


class GamePredictor:
    """
    Ensemble model predicting NBA game outcomes.

    Win Probability: Weighted ensemble of XGBoost + Random Forest + Logistic Regression
    Score Prediction: XGBoost regressor trained on point differentials + team averages
    """

    def __init__(self):
        self.win_model_xgb   = None
        self.win_model_rf    = None
        self.win_model_lr    = None
        self.score_model_home= None
        self.score_model_away= None
        self.scaler          = None
        self.is_trained      = False
        self.feature_importances_: Optional[pd.Series] = None

        # Ensemble weights
        self._weights = {"xgb": 0.55, "rf": 0.30, "lr": 0.15}

    # ─── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y_win: pd.Series,
        y_home_score: Optional[pd.Series] = None,
        y_away_score: Optional[pd.Series] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train all models on historical game data.
        Returns cross-validated accuracy metrics.
        """
        if not ML_AVAILABLE:
            logger.error("Cannot train: ML libraries not installed.")
            return {}

        if X.empty or len(y_win) == 0:
            logger.error("Empty training data.")
            return {}

        # Ensure consistent columns
        X = self._align_features(X)
        X_arr = X.values.astype(np.float32)

        logger.info(f"Training on {len(X_arr)} samples with {X_arr.shape[1]} features...")

        # ── Scaler ────────────────────────────────────────────────────────────
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X_arr)

        # ── XGBoost classifier ────────────────────────────────────────────────
        params = {k: v for k, v in XGBOOST_PARAMS.items()
                  if k not in ("use_label_encoder",)}
        self.win_model_xgb = xgb.XGBClassifier(**params)
        self.win_model_xgb.fit(X_arr, y_win,
                                eval_set=[(X_arr, y_win)], verbose=False)

        # ── Random Forest classifier ──────────────────────────────────────────
        self.win_model_rf = RandomForestClassifier(**RF_PARAMS)
        self.win_model_rf.fit(X_scaled, y_win)

        # ── Logistic Regression (calibrated) ─────────────────────────────────
        lr_base = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        self.win_model_lr = CalibratedClassifierCV(lr_base, cv=5, method="isotonic")
        self.win_model_lr.fit(X_scaled, y_win)

        # ── Cross-validation accuracy ─────────────────────────────────────────
        xgb_cv = cross_val_score(
            xgb.XGBClassifier(**params), X_arr, y_win, cv=5, scoring="accuracy"
        ).mean()
        rf_cv = cross_val_score(
            RandomForestClassifier(**RF_PARAMS), X_scaled, y_win, cv=5, scoring="accuracy"
        ).mean()

        # ── Score regressors ──────────────────────────────────────────────────
        if y_home_score is not None and len(y_home_score) == len(X_arr):
            score_params = {k: v for k, v in SCORE_MODEL_PARAMS.items()}
            self.score_model_home = xgb.XGBRegressor(**score_params)
            self.score_model_away = xgb.XGBRegressor(**score_params)
            self.score_model_home.fit(X_arr, y_home_score.values)
            self.score_model_away.fit(X_arr, y_away_score.values
                                      if y_away_score is not None
                                      else y_home_score.values - 5)
        else:
            # Train simpler score model from win labels + point diff proxy
            self.score_model_home = xgb.XGBRegressor(**SCORE_MODEL_PARAMS)
            self.score_model_away = xgb.XGBRegressor(**SCORE_MODEL_PARAMS)
            # Synthetic scores: average ~112 pts, win by ~6
            y_h = 112.0 + y_win.values * 3.0
            y_a = 112.0 - y_win.values * 3.0
            self.score_model_home.fit(X_arr, y_h)
            self.score_model_away.fit(X_arr, y_a)

        # ── Feature importances ───────────────────────────────────────────────
        self.feature_importances_ = pd.Series(
            self.win_model_xgb.feature_importances_,
            index=FEATURE_COLUMNS[:len(self.win_model_xgb.feature_importances_)]
        ).sort_values(ascending=False)

        self.is_trained = True
        metrics = {"xgb_cv_acc": round(xgb_cv, 4), "rf_cv_acc": round(rf_cv, 4)}
        if verbose:
            logger.info(f"Training complete. XGB CV: {xgb_cv:.3f}, RF CV: {rf_cv:.3f}")
        return metrics

    # ─── Prediction ────────────────────────────────────────────────────────────

    def predict_game(
        self,
        matchup_features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Predict win probabilities and scores for a single game.

        Args:
            matchup_features: Output of features.build_matchup_features()

        Returns:
            home_win_prob   : float 0-1
            away_win_prob   : float 0-1
            predicted_home_score: int
            predicted_away_score: int
            confidence      : "Low" | "Medium" | "High"
            margin          : predicted point differential (home - away)
        """
        if not self.is_trained:
            logger.warning("Model not trained. Loading from disk or using fallback.")
            if not self.load():
                return self._fallback_prediction(matchup_features)

        try:
            x = self._features_to_array(matchup_features)
            x_arr    = x.reshape(1, -1)
            x_scaled = self.scaler.transform(x_arr)

            # ── Win probability (ensemble) ─────────────────────────────────
            p_xgb = float(self.win_model_xgb.predict_proba(x_arr)[0, 1])
            p_rf  = float(self.win_model_rf.predict_proba(x_scaled)[0, 1])
            p_lr  = float(self.win_model_lr.predict_proba(x_scaled)[0, 1])

            home_win_prob = (
                self._weights["xgb"] * p_xgb +
                self._weights["rf"]  * p_rf +
                self._weights["lr"]  * p_lr
            )
            home_win_prob = float(np.clip(home_win_prob, 0.02, 0.98))
            away_win_prob = 1.0 - home_win_prob

            # ── Score prediction ───────────────────────────────────────────
            raw_home = float(self.score_model_home.predict(x_arr)[0])
            raw_away = float(self.score_model_away.predict(x_arr)[0])

            # Adjust scores based on win probability and injury impact
            home_inj = matchup_features.get("injury_impact_score", 0.0)
            away_inj = matchup_features.get("opp_injury_impact_score", 0.0)

            raw_home -= matchup_features.get("pts_lost_due_to_inj", 0.0)
            raw_away -= matchup_features.get("opp_pts_lost_due_to_inj", 0.0)

            # Clip to realistic range
            pred_home = int(round(np.clip(raw_home, 85, 145)))
            pred_away = int(round(np.clip(raw_away, 85, 145)))

            # Force consistency with win probability
            if home_win_prob > 0.55 and pred_home < pred_away:
                pred_home, pred_away = pred_away, pred_home
            elif home_win_prob < 0.45 and pred_home > pred_away:
                pred_home, pred_away = pred_away, pred_home

            margin = pred_home - pred_away

            # ── Confidence ────────────────────────────────────────────────
            spread = abs(home_win_prob - 0.5)
            if spread > 0.25:
                confidence = "High"
            elif spread > 0.12:
                confidence = "Medium"
            else:
                confidence = "Low"

            # ── Model agreement ───────────────────────────────────────────
            model_probs = [p_xgb, p_rf, p_lr]
            agreement   = 1.0 - np.std(model_probs) * 4  # 0-1 scale

            return {
                "home_win_prob":          round(home_win_prob, 4),
                "away_win_prob":          round(away_win_prob, 4),
                "predicted_home_score":   pred_home,
                "predicted_away_score":   pred_away,
                "margin":                 margin,
                "confidence":             confidence,
                "model_agreement":        round(float(agreement), 4),
                "xgb_prob":               round(p_xgb, 4),
                "rf_prob":                round(p_rf,  4),
                "lr_prob":                round(p_lr,  4),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(matchup_features)

    # ─── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODELS_DIR / "game_predictor.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or MODELS_DIR / "game_predictor.pkl"
        if not path.exists():
            logger.warning(f"No saved model found at {path}")
            return False
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
            logger.info("Model loaded from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature columns are in the correct order."""
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        for c in missing:
            X[c] = 0.0
        return X[FEATURE_COLUMNS]

    def _features_to_array(self, features: Dict) -> np.ndarray:
        return np.array(
            [float(features.get(col, 0.0)) for col in FEATURE_COLUMNS],
            dtype=np.float32
        )

    def _fallback_prediction(self, matchup: Dict) -> Dict:
        """
        Rule-based fallback when the ML model is not available.
        Uses net rating differential as the primary signal.
        """
        net_diff = matchup.get("net_rating_diff", 0.0)
        inj_diff = matchup.get("injury_diff", 0.0)  # + means away team more injured
        rest_diff= matchup.get("rest_diff", 0.0)     # + means home team more rested

        # Logistic transform
        logit = 0.15 + net_diff * 0.03 + inj_diff * 0.08 + rest_diff * 0.02
        home_wp = 1 / (1 + np.exp(-logit))
        home_wp = float(np.clip(home_wp, 0.10, 0.90))

        home_pts_base = matchup.get("pts_per_game", 112.0)
        away_pts_base = matchup.get("opp_pts_per_game", 112.0)
        pred_home = int(round(np.clip(home_pts_base - matchup.get("pts_lost_due_to_inj", 0), 90, 140)))
        pred_away = int(round(np.clip(away_pts_base - matchup.get("opp_pts_lost_due_to_inj", 0), 90, 140)))

        if home_wp > 0.55 and pred_home < pred_away:
            pred_home, pred_away = pred_away, pred_home

        return {
            "home_win_prob":        round(home_wp, 4),
            "away_win_prob":        round(1 - home_wp, 4),
            "predicted_home_score": pred_home,
            "predicted_away_score": pred_away,
            "margin":               pred_home - pred_away,
            "confidence":           "Low",
            "model_agreement":      0.6,
            "xgb_prob": round(home_wp, 4),
            "rf_prob":  round(home_wp, 4),
            "lr_prob":  round(home_wp, 4),
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top N most important features."""
        if self.feature_importances_ is None:
            return pd.DataFrame()
        return self.feature_importances_.head(top_n).reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        )
