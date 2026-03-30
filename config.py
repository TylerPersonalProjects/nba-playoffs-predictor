"""
NBA Playoffs Predictor - Configuration
"""
import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "cache"
MODELS_DIR = BASE_DIR / "models" / "trained"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── NBA Season ───────────────────────────────────────────────────────────────
CURRENT_SEASON       = "2024-25"
SEASON_TYPE_REGULAR  = "Regular Season"
SEASON_TYPE_PLAYOFFS = "Playoffs"

# ─── NBA Teams (abbreviation → full name) ─────────────────────────────────────
NBA_TEAMS = {
    "ATL": "Atlanta Hawks",      "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",      "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",      "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",   "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",        "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",  "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans","NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder","ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",  "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",          "WAS": "Washington Wizards",
}

# Conference alignment
EASTERN_TEAMS = ["ATL","BOS","BKN","CHA","CHI","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS","CLE"]
WESTERN_TEAMS = ["DAL","DEN","GSW","HOU","LAC","LAL","MEM","MIN","NOP","OKC","PHX","POR","SAC","SAS","UTA"]

# ─── Model Parameters ─────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth":    8,
    "random_state": 42,
    "n_jobs":       -1,
}

# Score regression model
SCORE_MODEL_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
}

# ─── Feature Columns ──────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Offensive metrics
    "off_rating", "pts_per_game", "fg_pct", "fg3_pct", "ft_pct",
    "ast_per_game", "oreb_per_game", "efg_pct", "ts_pct",
    # Defensive metrics
    "def_rating", "stl_per_game", "blk_per_game", "dreb_per_game",
    "opp_fg_pct", "opp_fg3_pct",
    # Net metrics
    "net_rating", "pace", "srs",
    # Form & situational
    "last10_win_pct", "rest_days", "is_home",
    # Head-to-head
    "h2h_win_pct", "h2h_avg_margin",
    # Injury
    "injury_impact_score",
    # Playoff experience
    "playoff_exp_score",
    # Opponent metrics (opponent's versions of all above)
    "opp_off_rating", "opp_pts_per_game", "opp_fg_pct_off", "opp_fg3_pct_off",
    "opp_ast_per_game", "opp_oreb_per_game", "opp_efg_pct", "opp_ts_pct",
    "opp_def_rating", "opp_stl_per_game", "opp_blk_per_game", "opp_dreb_per_game",
    "opp_net_rating", "opp_pace", "opp_srs",
    "opp_last10_win_pct", "opp_rest_days",
    "opp_injury_impact_score", "opp_playoff_exp_score",
    # Differential features (team - opponent)
    "net_rating_diff", "off_rating_diff", "def_rating_diff",
    "pts_per_game_diff", "fg_pct_diff", "injury_diff", "rest_diff",
    # ── Betting Market Features (Odds Shark + Action Network) ─────────────────
    "market_win_prob",        # Implied win prob from moneyline (vig-removed)
    "opp_market_win_prob",    # Opponent implied win prob
    "market_spread",          # Point spread (negative = this team favored)
    "market_total",           # Over/under total
    "public_bet_pct",         # Fraction of public bets on this team (0-1)
    "public_money_pct",       # Fraction of public money on this team (0-1)
    "sharp_money_indicator",  # money_pct - bet_pct (positive = sharp on team)
    "line_movement",          # Opening spread minus current spread
    "ats_pct",                # Team's ATS win % this season
    "opp_ats_pct",            # Opponent's ATS win % this season
    "ou_over_pct",            # Team tendency to go over (offensive proxy)
    "market_edge",            # ML model prob minus market implied prob
]

# Odds-specific feature columns (used to gracefully degrade when unavailable)
ODDS_FEATURE_COLUMNS = [
    "market_win_prob", "opp_market_win_prob", "market_spread", "market_total",
    "public_bet_pct", "public_money_pct", "sharp_money_indicator",
    "line_movement", "ats_pct", "opp_ats_pct", "ou_over_pct", "market_edge",
]

# ─── Injury Impact Weights ────────────────────────────────────────────────────
INJURY_STATUS_WEIGHTS = {
    "Out":         1.0,
    "Doubtful":    0.75,
    "Questionable":0.40,
    "Day-To-Day":  0.20,
    "Probable":    0.05,
}

# Star player usage weight (0-1 based on role)
PLAYER_ROLE_WEIGHTS = {
    "star":    1.0,
    "starter": 0.5,
    "bench":   0.2,
}

# ─── Data Refresh Settings ────────────────────────────────────────────────────
CACHE_TTL_SECONDS       = 300     # 5 min for live data
INJURY_REFRESH_SECONDS  = 180     # 3 min for injuries
STANDINGS_REFRESH_HOURS = 1       # 1 hour for standings

# ─── ESPN Injury Scrape URL ───────────────────────────────────────────────────
ESPN_INJURY_URL = "https://www.espn.com/nba/injuries"

# ─── Odds Data Sources ────────────────────────────────────────────────────────
ODDS_SHARK_ODDS_URL  = "https://www.oddsshark.com/nba/odds"
ODDS_SHARK_ATS_URL   = "https://www.oddsshark.com/nba/ats-standings"
ACTION_NETWORK_API   = "https://api.actionnetwork.com/web/v1/games"
ODDS_REFRESH_SECONDS = 180    # 3-minute cache for live odds
ATS_REFRESH_HOURS    = 1      # hourly ATS record refresh

# ─── Historical Seasons (for training) ───────────────────────────────────────
TRAINING_SEASONS = [
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"
]

# ─── Dashboard Settings ───────────────────────────────────────────────────────
DASHBOARD_TITLE     = "🏀 NBA Playoffs Predictor"
REFRESH_INTERVAL_MS = 300_000  # 5 minutes auto-refresh
PRIMARY_COLOR       = "#17408B"   # NBA blue
SECONDARY_COLOR     = "#C9082A"   # NBA red
ACCENT_COLOR        = "#F7B731"   # Gold
