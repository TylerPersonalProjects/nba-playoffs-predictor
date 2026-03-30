"""
Model Training Script
Fetches historical NBA game data and trains the ensemble prediction model.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --seasons 2022-23 2023-24 --verbose
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import TRAINING_SEASONS, MODELS_DIR, CURRENT_SEASON
from data.nba_data import get_historical_games, get_team_stats, get_player_stats
from data.features import build_training_features
from models.predictor import GamePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train(seasons: list = None, verbose: bool = True) -> None:
    seasons = seasons or TRAINING_SEASONS
    logger.info(f"Starting training pipeline. Seasons: {seasons}")

    # ── 1. Fetch historical games ──────────────────────────────────────────────
    logger.info("Fetching historical game data...")
    games_df = get_historical_games(seasons)
    logger.info(f"Loaded {len(games_df)} game records")

    if games_df.empty:
        logger.error("No game data found. Exiting.")
        sys.exit(1)

    # ── 2. Fetch team stats for each season ───────────────────────────────────
    logger.info("Fetching team stats by season...")
    team_stats_by_season = {}
    for season in seasons:
        try:
            ts = get_team_stats(season=season)
            if not ts.empty:
                team_stats_by_season[season] = ts
                logger.info(f"  {season}: {len(ts)} teams")
        except Exception as e:
            logger.warning(f"  Could not get stats for {season}: {e}")

    if not team_stats_by_season:
        logger.warning("No season stats. Using current season only.")
        team_stats_by_season = {CURRENT_SEASON: get_team_stats()}

    # ── 3. Build feature matrix ───────────────────────────────────────────────
    logger.info("Building feature matrix...")
    X, y_win, y_margin = build_training_features(games_df, team_stats_by_season)

    if X.empty:
        logger.error("Feature matrix is empty. Check data pipeline.")
        sys.exit(1)

    logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"Class balance: {y_win.mean():.1%} wins")

    # Handle NaNs
    X = X.fillna(0.0)

    # ── 4. Train model ────────────────────────────────────────────────────────
    logger.info("Training ensemble model...")
    predictor = GamePredictor()

    # Approximate home/away scores from margin
    y_home_score = 112.0 + y_margin / 2
    y_away_score = 112.0 - y_margin / 2

    metrics = predictor.train(
        X=X,
        y_win=y_win,
        y_home_score=y_home_score,
        y_away_score=y_away_score,
        verbose=verbose,
    )

    # ── 5. Report ─────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Training Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    if predictor.feature_importances_ is not None:
        logger.info("\nTop 10 Most Important Features:")
        top10 = predictor.feature_importances_.head(10)
        for feat, imp in top10.items():
            logger.info(f"  {feat:35s}  {imp:.4f}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    predictor.save()
    logger.info(f"\n✅ Model saved to {MODELS_DIR / 'game_predictor.pkl'}")
    logger.info("Run `streamlit run dashboard/app.py` to start the dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA Playoffs Predictor")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Season strings, e.g. 2022-23 2023-24")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    train(seasons=args.seasons, verbose=args.verbose)
