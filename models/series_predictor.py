"""
Playoff Series Predictor
Simulates best-of-7 NBA playoff series using the GamePredictor.
Tracks live series state and updates predictions after every game.
"""

import logging
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.predictor import GamePredictor
from data.features import build_matchup_features
from data.nba_data import get_team_stats, get_player_stats
from data.injury_tracker import get_team_injury_summary

logger = logging.getLogger(__name__)

# Playoff series format: 2-2-1-1-1 home games
HOME_GAME_SEQUENCE = {
    "high_seed": [1, 1, 0, 0, 1, 0, 1],  # 1=high seed home, 0=low seed home
}

# Realistic rest-day distributions between playoff games
REST_DAYS_PATTERN = [2, 2, 1, 2, 2, 2, 2]


class SeriesState:
    """Tracks the state of a single playoff series."""

    def __init__(self, team_high: str, team_low: str):
        self.team_high     = team_high    # Higher-seeded / home team (Games 1-2, 5, 7)
        self.team_low      = team_low     # Lower-seeded team
        self.wins_high     = 0
        self.wins_low      = 0
        self.game_results  = []           # List of (winning_team, home_score, away_score)
        self.completed     = False
        self.winner        = None
        self.game_number   = 1

    @property
    def series_label(self) -> str:
        return f"{self.wins_high}-{self.wins_low}"

    @property
    def games_played(self) -> int:
        return self.wins_high + self.wins_low

    def record_game(self, winner: str, high_score: int, low_score: int) -> None:
        """Record the result of a game and check if series is over."""
        if winner == self.team_high:
            self.wins_high += 1
            self.game_results.append({
                "game": self.game_number,
                "winner": winner,
                "high_score": high_score,
                "low_score": low_score,
                "home_team": self.get_home_team(),
            })
        else:
            self.wins_low += 1
            self.game_results.append({
                "game": self.game_number,
                "winner": winner,
                "high_score": high_score,
                "low_score": low_score,
                "home_team": self.get_home_team(),
            })

        self.game_number += 1

        if self.wins_high == 4:
            self.completed = True
            self.winner    = self.team_high
        elif self.wins_low == 4:
            self.completed = True
            self.winner    = self.team_low

    def get_home_team(self) -> str:
        """Return which team is home for the current game number."""
        idx = (self.game_number - 1) % 7
        return (
            self.team_high
            if HOME_GAME_SEQUENCE["high_seed"][idx] == 1
            else self.team_low
        )

    def to_dict(self) -> Dict:
        return {
            "team_high":    self.team_high,
            "team_low":     self.team_low,
            "wins_high":    self.wins_high,
            "wins_low":     self.wins_low,
            "series_label": self.series_label,
            "completed":    self.completed,
            "winner":       self.winner,
            "games_played": self.games_played,
            "game_results": self.game_results,
        }


class SeriesPredictor:
    """
    Predicts the outcome of a full NBA playoff series (best-of-7).

    Capabilities:
        - Predict series win probability for each team
        - Predict likely series length (4, 5, 6, or 7 games)
        - Predict score for each individual game
        - Simulate thousands of series to get probability distribution
        - Update predictions live as games are played
    """

    def __init__(self, game_predictor: Optional[GamePredictor] = None):
        self.predictor = game_predictor or GamePredictor()
        if not self.predictor.is_trained:
            self.predictor.load()

    def predict_series(
        self,
        team_high: str,
        team_low: str,
        series_state: Optional[SeriesState] = None,
        n_simulations: int = 10000,
        home_court_boost: float = 0.035,
    ) -> Dict:
        """
        Predict the outcome of a playoff series via Monte Carlo simulation.

        Args:
            team_high       : High seed (home court advantage)
            team_low        : Low seed
            series_state    : Current series state (if mid-series)
            n_simulations   : Number of Monte Carlo simulations
            home_court_boost: Additional win prob for home team

        Returns:
            Comprehensive series prediction dict.
        """
        # Get current data
        team_stats   = get_team_stats()
        player_stats = get_player_stats()
        inj_summary  = get_team_injury_summary()

        # Build base matchup features (Game 1: high seed at home)
        matchup = build_matchup_features(
            home_team=team_high,
            away_team=team_low,
            team_stats_df=team_stats,
            player_stats_df=player_stats,
            injury_summary=inj_summary,
            home_rest_days=2,
            away_rest_days=2,
        )

        # Base win probabilities from model
        base_pred = self.predictor.predict_game(matchup)
        base_high_wp = base_pred["home_win_prob"]

        # Apply home court boost
        high_home_wp = min(base_high_wp + home_court_boost, 0.95)
        high_away_wp = max(base_high_wp - home_court_boost, 0.05)
        low_home_wp  = 1.0 - high_away_wp
        low_away_wp  = 1.0 - high_home_wp

        # Current state
        if series_state is None:
            series_state = SeriesState(team_high, team_low)

        wins_h = series_state.wins_high
        wins_l = series_state.wins_low

        # Monte Carlo simulation
        high_series_wins = 0
        series_lengths   = []
        game_log         = []

        for _ in range(n_simulations):
            h, l = wins_h, wins_l
            for game_num in range(series_state.game_number, 8):
                idx     = (game_num - 1) % 7
                is_high_home = HOME_GAME_SEQUENCE["high_seed"][idx] == 1
                wp_h    = high_home_wp if is_high_home else high_away_wp

                # Injury adjustment per game
                wp_h *= (1 - matchup.get("injury_impact_score", 0.0) * 0.15)
                wp_h += matchup.get("opp_injury_impact_score", 0.0) * 0.15
                wp_h  = np.clip(wp_h, 0.05, 0.95)

                if random.random() < wp_h:
                    h += 1
                else:
                    l += 1

                if h == 4:
                    high_series_wins += 1
                    series_lengths.append(game_num)
                    break
                elif l == 4:
                    series_lengths.append(game_num)
                    break

        high_win_pct = high_series_wins / n_simulations
        low_win_pct  = 1.0 - high_win_pct

        # Predicted series length distribution
        if series_lengths:
            from collections import Counter
            length_dist = Counter(series_lengths)
            most_likely_length = max(length_dist, key=length_dist.get)
            length_probs = {k: round(v / n_simulations, 3)
                            for k, v in sorted(length_dist.items())}
        else:
            most_likely_length = 6
            length_probs = {4: 0.10, 5: 0.25, 6: 0.35, 7: 0.30}

        # Per-game predictions for remaining games
        remaining_games = self._predict_remaining_games(
            team_high, team_low, series_state,
            matchup, base_pred,
            high_home_wp, high_away_wp,
            team_stats, player_stats, inj_summary,
        )

        # SHAP / key factors
        key_factors = self._build_key_factors(matchup, base_pred)

        return {
            "team_high":           team_high,
            "team_low":            team_low,
            "high_win_probability": round(high_win_pct, 4),
            "low_win_probability":  round(low_win_pct, 4),
            "predicted_winner":    team_high if high_win_pct >= 0.5 else team_low,
            "predicted_length":    most_likely_length,
            "length_distribution": length_probs,
            "current_series":      f"{wins_h}-{wins_l}",
            "games_played":        wins_h + wins_l,
            "per_game_predictions":remaining_games,
            "key_factors":         key_factors,
            "base_game1_pred":     base_pred,
            "injury_impact_high":  matchup.get("injury_impact_score", 0.0),
            "injury_impact_low":   matchup.get("opp_injury_impact_score", 0.0),
            "simulations_run":     n_simulations,
        }

    def _predict_remaining_games(
        self,
        team_high: str,
        team_low: str,
        series_state: SeriesState,
        base_matchup: Dict,
        base_pred: Dict,
        high_home_wp: float,
        high_away_wp: float,
        team_stats,
        player_stats,
        inj_summary,
    ) -> List[Dict]:
        """Predict each of the remaining possible games in the series."""
        predictions = []
        wins_h = series_state.wins_high
        wins_l = series_state.wins_low

        for game_num in range(series_state.game_number, 8):
            if wins_h == 4 or wins_l == 4:
                break

            idx          = (game_num - 1) % 7
            is_high_home = HOME_GAME_SEQUENCE["high_seed"][idx] == 1
            home_team    = team_high if is_high_home else team_low
            away_team    = team_low  if is_high_home else team_high
            wp_high      = high_home_wp if is_high_home else high_away_wp

            # Build matchup for this specific game
            game_matchup = build_matchup_features(
                home_team=home_team,
                away_team=away_team,
                team_stats_df=team_stats,
                player_stats_df=player_stats,
                injury_summary=inj_summary,
                home_rest_days=REST_DAYS_PATTERN[game_num - 1],
                away_rest_days=REST_DAYS_PATTERN[game_num - 1],
            )
            game_pred = self.predictor.predict_game(game_matchup)

            predictions.append({
                "game_number":   game_num,
                "home_team":     home_team,
                "away_team":     away_team,
                "series_score":  f"{wins_h}-{wins_l}",
                "high_win_prob": round(wp_high, 3),
                "home_win_prob": round(game_pred["home_win_prob"], 3),
                "pred_home_score": game_pred["predicted_home_score"],
                "pred_away_score": game_pred["predicted_away_score"],
                "pred_margin":   game_pred["margin"],
                "confidence":    game_pred["confidence"],
                "location":      f"{home_team} (Home)",
            })

            # Simulate most likely outcome for series progression
            if wp_high >= 0.5:
                wins_h += 1
            else:
                wins_l += 1

        return predictions

    def _build_key_factors(self, matchup: Dict, pred: Dict) -> List[Dict]:
        """Extract the top factors driving the series prediction."""
        factors = []

        net_diff = matchup.get("net_rating_diff", 0)
        if abs(net_diff) > 2:
            team = "home team" if net_diff > 0 else "away team"
            factors.append({
                "factor": "Net Rating Advantage",
                "detail": f"{team.title()} has a +{abs(net_diff):.1f} net rating edge",
                "impact": "High",
            })

        inj_h = matchup.get("injury_impact_score", 0)
        inj_a = matchup.get("opp_injury_impact_score", 0)
        if max(inj_h, inj_a) > 0.2:
            more_inj = "Home team" if inj_h > inj_a else "Away team"
            factors.append({
                "factor": "Injury Impact",
                "detail": f"{more_inj} facing significant injury concerns",
                "impact": "High" if max(inj_h, inj_a) > 0.4 else "Medium",
            })

        h2h = matchup.get("h2h_win_pct", 0.5)
        if abs(h2h - 0.5) > 0.15:
            dom = "home" if h2h > 0.5 else "away"
            factors.append({
                "factor": "Head-to-Head History",
                "detail": f"{dom.title()} team owns {h2h:.0%} H2H win rate this season",
                "impact": "Medium",
            })

        form = matchup.get("last10_win_pct", 0.5)
        if form > 0.75 or form < 0.30:
            label = "hot" if form > 0.75 else "cold"
            factors.append({
                "factor": "Recent Form",
                "detail": f"Home team is {label} ({form:.0%} last 10 games)",
                "impact": "Medium",
            })

        pace_diff = abs(matchup.get("pace", 99) - matchup.get("opp_pace", 99))
        if pace_diff > 3:
            factors.append({
                "factor": "Pace Mismatch",
                "detail": f"Significant pace differential ({pace_diff:.1f} possessions) could affect style",
                "impact": "Low",
            })

        if not factors:
            factors.append({
                "factor": "Evenly Matched",
                "detail": "Both teams are closely matched — series could go either way",
                "impact": "Low",
            })

        return factors

    def update_series_prediction(
        self,
        series: "PlayoffSeries",
        game_result: Dict,
    ) -> Dict:
        """
        Update series predictions after a game result is recorded.
        Called automatically by the real-time update loop.
        """
        series.state.record_game(
            winner      = game_result["winner"],
            high_score  = game_result.get("high_score", 110),
            low_score   = game_result.get("low_score", 105),
        )

        if series.state.completed:
            return {
                "completed": True,
                "winner":    series.state.winner,
                "final_record": series.state.series_label,
            }

        return self.predict_series(
            team_high    = series.state.team_high,
            team_low     = series.state.team_low,
            series_state = series.state,
        )


class PlayoffBracket:
    """
    Manages the full NBA playoff bracket with all series predictions.
    Tracks results in real-time and propagates winners to next rounds.
    """

    BRACKET_STRUCTURE = {
        "East": {
            "R1": [
                ("1E", "8E"), ("2E", "7E"), ("3E", "6E"), ("4E", "5E")
            ],
        },
        "West": {
            "R1": [
                ("1W", "8W"), ("2W", "7W"), ("3W", "6W"), ("4W", "5W")
            ],
        },
    }

    def __init__(
        self,
        seedings: Dict[str, str],  # e.g. {"1E": "Boston Celtics", "8E": "Miami Heat", ...}
        game_predictor: Optional[GamePredictor] = None,
    ):
        self.seedings     = seedings
        self.series_pred  = SeriesPredictor(game_predictor)
        self.series       = {}      # series_id → SeriesState
        self.predictions  = {}     # series_id → prediction dict
        self._initialize_bracket()

    def _initialize_bracket(self) -> None:
        """Build initial Round 1 series."""
        for conf, rounds in self.BRACKET_STRUCTURE.items():
            for matchup in rounds["R1"]:
                high_seed, low_seed = matchup
                high_team = self.seedings.get(high_seed, high_seed)
                low_team  = self.seedings.get(low_seed, low_seed)
                series_id = f"{high_seed}_vs_{low_seed}"
                self.series[series_id]      = SeriesState(high_team, low_team)
                self.predictions[series_id] = {}

    def predict_all_series(self, n_simulations: int = 5000) -> Dict:
        """Generate predictions for all active series."""
        for series_id, state in self.series.items():
            if not state.completed:
                try:
                    pred = self.series_pred.predict_series(
                        team_high    = state.team_high,
                        team_low     = state.team_low,
                        series_state = state,
                        n_simulations= n_simulations,
                    )
                    self.predictions[series_id] = pred
                except Exception as e:
                    logger.error(f"Error predicting {series_id}: {e}")
        return self.predictions

    def predict_full_bracket(self, n_simulations: int = 10000) -> Dict:
        """
        Simulate the entire bracket from current state to champion.
        Returns win probabilities for each team to win the championship.
        """
        champion_counts = {}

        for _ in range(n_simulations):
            sim_series = deepcopy(self.series)
            sim_results = {}

            # Simulate Round 1
            r1_winners = self._simulate_round(sim_series, n_games=7)

            # Build R2 matchups (1 vs 4 winner, 2 vs 3 winner)
            r2_series = self._build_next_round(r1_winners)
            r2_winners = self._simulate_round(r2_series, n_games=7)

            # Conf Finals
            cf_series = self._build_next_round(r2_winners)
            cf_winners = self._simulate_round(cf_series, n_games=7)

            # NBA Finals
            if len(cf_winners) >= 2:
                finalist_teams = list(cf_winners.values())
                if len(finalist_teams) == 2:
                    champ = self._simulate_single_series(
                        finalist_teams[0], finalist_teams[1]
                    )
                    champion_counts[champ] = champion_counts.get(champ, 0) + 1

        total = sum(champion_counts.values())
        return {
            team: round(count / total, 4)
            for team, count in sorted(champion_counts.items(),
                                      key=lambda x: -x[1])
        } if total > 0 else {}

    def _simulate_round(self, series_dict: Dict, n_games: int = 7) -> Dict:
        """Simulate one round of the bracket, return winners."""
        winners = {}
        for sid, state in series_dict.items():
            winner = self._simulate_single_series(state.team_high, state.team_low)
            winners[sid] = winner
        return winners

    def _simulate_single_series(self, team_h: str, team_l: str) -> str:
        """Fast single-series simulation using base win probabilities."""
        try:
            matchup = build_matchup_features(team_h, team_l)
            pred = self.series_pred.predictor.predict_game(matchup)
            wp_h = float(pred.get("home_win_prob", 0.5))
        except Exception:
            wp_h = 0.5

        wins_h = wins_l = 0
        for gn in range(7):
            home_is_high = HOME_GAME_SEQUENCE["high_seed"][gn] == 1
            wp = min(wp_h + 0.035, 0.95) if home_is_high else max(wp_h - 0.035, 0.05)
            if random.random() < wp:
                wins_h += 1
            else:
                wins_l += 1
            if wins_h == 4:
                return team_h
            if wins_l == 4:
                return team_l
        return team_h if wins_h > wins_l else team_l

    def _build_next_round(self, winners: Dict) -> Dict:
        """Stub: Build next round series from winners dict."""
        teams = list(winners.values())
        new_series = {}
        for i in range(0, len(teams) - 1, 2):
            sid = f"next_{i}"
            new_series[sid] = SeriesState(teams[i], teams[i+1])
        return new_series

    def get_bracket_summary(self) -> Dict:
        """Return a serializable summary of the entire bracket."""
        return {
            "seedings":    self.seedings,
            "series":      {k: v.to_dict() for k, v in self.series.items()},
            "predictions": self.predictions,
        }
