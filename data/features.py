"""
Feature Engineering Module
Builds the complete feature vector for each team/matchup used by the ML model.
Includes NBA stats, injury data, and live betting market signals from
Odds Shark and Action Network.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import FEATURE_COLUMNS, CURRENT_SEASON
from data.nba_data import (
    get_team_stats, get_player_stats, get_recent_form,
    get_head_to_head, get_all_nba_teams, _get_team_id,
)
from data.injury_tracker import get_team_injury_summary

logger = logging.getLogger(__name__)

# Lazy import odds scraper — avoids hard failure if requests lib unavailable
def _get_odds_features(team_abbr: str, opp_abbr: str, is_home: bool) -> Dict[str, float]:
    """Fetch odds-based features with graceful fallback."""
    defaults = {
        "market_win_prob": 0.5, "opp_market_win_prob": 0.5,
        "market_spread": 0.0, "market_total": 220.0,
        "public_bet_pct": 0.5, "public_money_pct": 0.5,
        "sharp_money_indicator": 0.0, "line_movement": 0.0,
        "ats_pct": 0.5, "opp_ats_pct": 0.5, "ou_over_pct": 0.5,
        "market_edge": 0.0,
    }
    try:
        from data.odds_scraper import get_odds_features_for_team
        return get_odds_features_for_team(team_abbr, opp_abbr, is_home)
    except Exception as e:
        logger.debug(f"Odds features unavailable ({e}), using defaults")
        return defaults


# ─── Build Team Feature Dict ──────────────────────────────────────────────────

def build_team_features(
    team_name: str,
    team_stats_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    injury_summary: Dict,
    is_home: bool = True,
    rest_days: int = 2,
) -> Dict[str, float]:
    """
    Build a complete feature dictionary for one team in a matchup.
    """
    # ── Find team in stats DF ──────────────────────────────────────────────────
    team_row = _find_team_row(team_name, team_stats_df)
    if team_row is None:
        logger.warning(f"No stats found for team: {team_name}")
        team_row = _default_team_row()

    # ── Core offensive / defensive / net metrics ───────────────────────────────
    features = {
        "off_rating":    _safe(team_row, "off_rating",    112.0),
        "def_rating":    _safe(team_row, "def_rating",    112.0),
        "net_rating":    _safe(team_row, "net_rating",    0.0),
        "pts_per_game":  _safe(team_row, "pts_per_game",  112.0),
        "fg_pct":        _safe(team_row, "fg_pct",        0.46),
        "fg3_pct":       _safe(team_row, "fg3_pct",       0.36),
        "ft_pct":        _safe(team_row, "ft_pct",        0.78),
        "ast_per_game":  _safe(team_row, "ast_per_game",  26.0),
        "oreb_per_game": _safe(team_row, "oreb_per_game", 10.5),
        "dreb_per_game": _safe(team_row, "dreb_per_game", 33.0),
        "stl_per_game":  _safe(team_row, "stl_per_game",  7.0),
        "blk_per_game":  _safe(team_row, "blk_per_game",  5.0),
        "tov_per_game":  _safe(team_row, "tov_per_game",  13.5),
        "efg_pct":       _safe(team_row, "efg_pct",       0.54),
        "ts_pct":        _safe(team_row, "ts_pct",        0.58),
        "pace":          _safe(team_row, "pace",           99.0),
        "srs":           _safe(team_row, "srs",            0.0),
        "win_pct":       _safe(team_row, "win_pct",        0.50),
        "opp_fg_pct":    _safe(team_row, "opp_fg_pct",    0.46),
        "opp_fg3_pct":   _safe(team_row, "opp_fg3_pct",   0.36),
    }

    # ── Recent form ────────────────────────────────────────────────────────────
    team_id = _safe(team_row, "team_id", None)
    if team_id and not np.isnan(float(team_id)):
        form = get_recent_form(int(team_id), n_games=10)
    else:
        form = {"last_n_win_pct": 0.5, "last_n_avg_pts": 112.0,
                "last_n_avg_margin": 0.0, "streak": 0}
    features["last10_win_pct"] = form["last_n_win_pct"]
    features["streak"]         = form["streak"]

    # ── Situational ────────────────────────────────────────────────────────────
    features["is_home"]   = float(is_home)
    features["rest_days"] = float(max(0, min(rest_days, 10)))

    # ── Injury impact ──────────────────────────────────────────────────────────
    inj = _find_injury(team_name, injury_summary)
    features["injury_impact_score"] = inj.get("impact_score", 0.0)
    features["pts_lost_due_to_inj"] = inj.get("pts_lost_per_game", 0.0)

    # ── Playoff experience score ───────────────────────────────────────────────
    features["playoff_exp_score"] = calculate_playoff_experience(
        team_name, player_stats_df
    )

    # ── Star player metrics ────────────────────────────────────────────────────
    star_metrics = get_star_player_metrics(team_name, player_stats_df)
    features.update(star_metrics)

    return features


# ─── Build Matchup Features ───────────────────────────────────────────────────

def build_matchup_features(
    home_team: str,
    away_team: str,
    team_stats_df: Optional[pd.DataFrame] = None,
    player_stats_df: Optional[pd.DataFrame] = None,
    injury_summary: Optional[Dict] = None,
    home_rest_days: int = 2,
    away_rest_days: int = 2,
) -> Dict[str, float]:
    """
    Build the full feature vector for a home vs. away matchup.
    This is the input fed into the ML model.
    """
    if team_stats_df is None:
        team_stats_df = get_team_stats()
    if player_stats_df is None:
        player_stats_df = get_player_stats()
    if injury_summary is None:
        injury_summary = get_team_injury_summary()

    home_feat = build_team_features(
        home_team, team_stats_df, player_stats_df, injury_summary,
        is_home=True, rest_days=home_rest_days
    )
    away_feat = build_team_features(
        away_team, team_stats_df, player_stats_df, injury_summary,
        is_home=False, rest_days=away_rest_days
    )

    # Head-to-head
    h2h = get_head_to_head(home_team, away_team)
    home_feat["h2h_win_pct"]    = h2h.get("h2h_win_pct",    0.5)
    home_feat["h2h_avg_margin"] = h2h.get("h2h_avg_margin", 0.0)
    away_feat["h2h_win_pct"]    = 1.0 - h2h.get("h2h_win_pct", 0.5)
    away_feat["h2h_avg_margin"] = -h2h.get("h2h_avg_margin", 0.0)

    # ── Betting market features (Odds Shark + Action Network) ─────────────────
    home_abbr = _team_abbr_from_name(home_team)
    away_abbr = _team_abbr_from_name(away_team)
    home_odds = _get_odds_features(home_abbr, away_abbr, is_home=True)
    away_odds = _get_odds_features(away_abbr, home_abbr, is_home=False)
    home_feat.update(home_odds)
    away_feat.update(away_odds)

    # Build differential features (home - away)
    matchup = {}
    for k, v in home_feat.items():
        matchup[k] = v
    for k, v in away_feat.items():
        matchup[f"opp_{k}"] = v

    # Key differentials
    matchup["net_rating_diff"]  = home_feat["net_rating"]  - away_feat["net_rating"]
    matchup["off_rating_diff"]  = home_feat["off_rating"]  - away_feat["def_rating"]
    matchup["def_rating_diff"]  = away_feat["off_rating"]  - home_feat["def_rating"]
    matchup["pts_per_game_diff"]= home_feat["pts_per_game"]- away_feat["pts_per_game"]
    matchup["fg_pct_diff"]      = home_feat["fg_pct"]      - away_feat["opp_fg_pct"]
    matchup["injury_diff"]      = away_feat["injury_impact_score"] - home_feat["injury_impact_score"]
    matchup["rest_diff"]        = home_feat["rest_days"]   - away_feat["rest_days"]

    return matchup


# ─── Features as Flat Array (for model input) ─────────────────────────────────

def matchup_to_array(matchup: Dict[str, float]) -> np.ndarray:
    """Convert matchup dict to numpy array in consistent column order."""
    return np.array([matchup.get(col, 0.0) for col in FEATURE_COLUMNS],
                    dtype=np.float32)


def build_training_features(
    games_df: pd.DataFrame,
    team_stats_by_season: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build training feature matrix from historical game logs.

    Returns:
        X: feature DataFrame
        y_win: binary win label
        y_margin: point differential (for score regression)
    """
    if games_df.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=float)

    if team_stats_by_season is None:
        team_stats_by_season = {CURRENT_SEASON: get_team_stats()}

    rows    = []
    y_win   = []
    y_margin = []

    for _, game in games_df.iterrows():
        try:
            season = str(game.get("season", CURRENT_SEASON))
            team_stats = team_stats_by_season.get(season, get_team_stats())
            player_stats = get_player_stats(season=season)
            injury_summary = {}  # No historical injuries

            home_away = str(game.get("MATCHUP", "@ "))
            is_home   = "vs." in home_away

            matchup = build_matchup_features(
                home_team=str(game.get("TEAM_ABBREVIATION", "")),
                away_team=str(game.get("OPPONENT_TEAM_ABBREVIATION", "")),
                team_stats_df=team_stats,
                player_stats_df=player_stats,
                injury_summary=injury_summary,
                home_rest_days=2,
                away_rest_days=2,
            )

            rows.append(matchup_to_array(matchup))
            y_win.append(int(game.get("WIN", game.get("WL") == "W")))
            y_margin.append(float(game.get("PLUS_MINUS", 0)))

        except Exception as e:
            logger.debug(f"Skipping game row due to error: {e}")
            continue

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=float)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return X, pd.Series(y_win), pd.Series(y_margin)


# ─── Star Player Metrics ──────────────────────────────────────────────────────

def get_star_player_metrics(
    team_name: str,
    player_stats_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract top-3 player metrics for a team.
    Returns star1_pts, star2_pts, star3_pts, star_avg_ts, star_avg_usg.
    """
    if player_stats_df.empty:
        return {f"star{i}_pts": 15.0 for i in range(1, 4)} | {
            "star_avg_ts": 0.57, "star_avg_usg": 0.22
        }

    team_players = player_stats_df[
        player_stats_df["team_abbr"].str.upper() == _team_abbr_from_name(team_name)
    ].sort_values("pts_per_game", ascending=False)

    result = {}
    for i in range(3):
        if i < len(team_players):
            row = team_players.iloc[i]
            result[f"star{i+1}_pts"]  = float(row.get("pts_per_game", 15))
            result[f"star{i+1}_reb"]  = float(row.get("reb_per_game", 5))
            result[f"star{i+1}_ast"]  = float(row.get("ast_per_game", 3))
            result[f"star{i+1}_ts"]   = float(row.get("ts_pct", 0.57))
        else:
            result[f"star{i+1}_pts"] = 10.0
            result[f"star{i+1}_reb"] = 3.0
            result[f"star{i+1}_ast"] = 2.0
            result[f"star{i+1}_ts"]  = 0.55

    if not team_players.empty:
        top3 = team_players.head(3)
        result["star_avg_ts"]  = float(top3["ts_pct"].mean()) if "ts_pct" in top3 else 0.57
        result["star_avg_usg"] = float(top3.get("usg_pct", pd.Series([0.22])).mean())
    else:
        result["star_avg_ts"]  = 0.57
        result["star_avg_usg"] = 0.22

    return result


# ─── Playoff Experience ───────────────────────────────────────────────────────

def calculate_playoff_experience(
    team_name: str,
    player_stats_df: pd.DataFrame,
) -> float:
    """
    Estimate playoff experience score (0-1) based on team composition.
    Veteran-heavy rosters tend to perform better in the playoffs.
    (Uses minutes played as a proxy for experience — veterans play more.)
    """
    if player_stats_df.empty:
        return 0.5

    team_players = player_stats_df[
        player_stats_df["team_abbr"].str.upper() == _team_abbr_from_name(team_name)
    ]

    if team_players.empty:
        return 0.5

    # Weight by minutes: more minutes → more experience assumed
    avg_min = team_players["min_per_game"].mean() if "min_per_game" in team_players else 24.0
    # Stars (high usage, high minutes) weight more
    score = float(np.clip((avg_min - 18) / 20, 0, 1))
    return round(score, 4)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _find_team_row(team_name: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """Find a team row in a stats DataFrame by name or abbreviation."""
    if df.empty:
        return None

    # Try exact match on team_name
    if "team_name" in df.columns:
        match = df[df["team_name"].str.lower() == team_name.lower()]
        if not match.empty:
            return match.iloc[0]

    # Try partial match
    if "team_name" in df.columns:
        match = df[df["team_name"].str.lower().str.contains(
            team_name.lower()[:5], na=False
        )]
        if not match.empty:
            return match.iloc[0]

    # Try abbreviation
    abbr = _team_abbr_from_name(team_name)
    if "team_abbr" in df.columns:
        match = df[df["team_abbr"].str.upper() == abbr]
        if not match.empty:
            return match.iloc[0]

    return None


def _find_injury(team_name: str, injury_summary: Dict) -> Dict:
    """Find injury data for a team, tolerating name variations."""
    for key in injury_summary:
        if team_name.lower() in key.lower() or key.lower() in team_name.lower():
            return injury_summary[key]
    return {"impact_score": 0.0, "pts_lost_per_game": 0.0}


def _safe(row, col: str, default):
    """Safely extract a value from a Series, returning default if missing/NaN."""
    try:
        v = row[col]
        if pd.isna(v):
            return default
        return float(v)
    except (KeyError, TypeError, ValueError):
        return default


def _default_team_row() -> pd.Series:
    """Return a default team stats row (league average)."""
    return pd.Series({
        "team_id": np.nan, "off_rating": 112.0, "def_rating": 112.0,
        "net_rating": 0.0, "pts_per_game": 112.0, "fg_pct": 0.46,
        "fg3_pct": 0.36, "ft_pct": 0.78, "ast_per_game": 26.0,
        "oreb_per_game": 10.5, "dreb_per_game": 33.0, "stl_per_game": 7.0,
        "blk_per_game": 5.0, "tov_per_game": 13.5, "efg_pct": 0.54,
        "ts_pct": 0.58, "pace": 99.0, "srs": 0.0, "win_pct": 0.50,
        "opp_fg_pct": 0.46, "opp_fg3_pct": 0.36,
    })


# ─── Name → Abbreviation Map ──────────────────────────────────────────────────
_NAME_TO_ABBR = {
    "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
    "charlotte hornets": "CHA", "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE", "dallas mavericks": "DAL",
    "denver nuggets": "DEN", "detroit pistons": "DET",
    "golden state warriors": "GSW", "houston rockets": "HOU",
    "indiana pacers": "IND", "la clippers": "LAC", "los angeles clippers": "LAC",
    "los angeles lakers": "LAL", "la lakers": "LAL",
    "memphis grizzlies": "MEM", "miami heat": "MIA",
    "milwaukee bucks": "MIL", "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP", "new york knicks": "NYK",
    "oklahoma city thunder": "OKC", "orlando magic": "ORL",
    "philadelphia 76ers": "PHI", "phoenix suns": "PHX",
    "portland trail blazers": "POR", "sacramento kings": "SAC",
    "san antonio spurs": "SAS", "toronto raptors": "TOR",
    "utah jazz": "UTA", "washington wizards": "WAS",
}


def _team_abbr_from_name(name: str) -> str:
    return _NAME_TO_ABBR.get(name.lower().strip(), name.upper()[:3])
