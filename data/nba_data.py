"""
NBA Data Collection Module
Fetches real-time and historical data from the NBA API.
All results are cached to disk to respect rate limits.
"""

import time
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np
import requests

try:
    from nba_api.stats.endpoints import (
        leaguedashteamstats,
        leaguedashplayerstats,
        teamgamelogs,
        leaguestandings,
        scoreboardv2,
        leaguegamefinder,
        playergamelogs,
        boxscoresummaryv2,
        leaguedashteamclutch,
        teamestimatedmetrics,
    )
    from nba_api.stats.static import teams as nba_teams_static
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    logging.warning("nba_api not installed. Install with: pip install nba_api")

from config import (
    DATA_DIR, CURRENT_SEASON, SEASON_TYPE_REGULAR, SEASON_TYPE_PLAYOFFS,
    CACHE_TTL_SECONDS, TRAINING_SEASONS
)

logger = logging.getLogger(__name__)

# nba_api rate limit — sleep between requests
API_SLEEP = 0.6  # seconds


# ─── Cache Helpers ────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return DATA_DIR / f"{key}.pkl"


def _load_cache(key: str, ttl: int = CACHE_TTL_SECONDS) -> Optional[Any]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(key: str, data: Any) -> None:
    path = _cache_path(key)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ─── Team Stats ───────────────────────────────────────────────────────────────

def get_team_stats(
    season: str = CURRENT_SEASON,
    season_type: str = SEASON_TYPE_REGULAR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch team-level statistics for a given season.
    Includes: points, FG%, 3P%, rebounds, assists, steals, blocks,
              offensive rating, defensive rating, net rating, pace, TS%, eFG%.
    """
    cache_key = f"team_stats_{season}_{season_type.replace(' ', '_')}"
    if not force_refresh:
        cached = _load_cache(cache_key, ttl=3600)
        if cached is not None:
            return cached

    if not NBA_API_AVAILABLE:
        return _mock_team_stats()

    try:
        # Base stats
        time.sleep(API_SLEEP)
        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Base",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]

        time.sleep(API_SLEEP)
        advanced = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]

        time.sleep(API_SLEEP)
        opponent = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Opponent",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]

        # Merge on TEAM_ID
        df = base.merge(
            advanced[["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING",
                       "AST_PCT", "AST_TO", "AST_RATIO", "OREB_PCT", "DREB_PCT",
                       "REB_PCT", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "PACE",
                       "PIE"]],
            on="TEAM_ID", how="left"
        ).merge(
            opponent[["TEAM_ID", "FG_PCT", "FG3_PCT", "FG3A"]].rename(
                columns={"FG_PCT": "OPP_FG_PCT", "FG3_PCT": "OPP_FG3_PCT", "FG3A": "OPP_FG3A"}
            ),
            on="TEAM_ID", how="left"
        )

        # Rename for consistency
        df = df.rename(columns={
            "TEAM_NAME":   "team_name",
            "TEAM_ID":     "team_id",
            "GP":          "games_played",
            "W":           "wins",
            "L":           "losses",
            "PTS":         "pts_per_game",
            "FG_PCT":      "fg_pct",
            "FG3_PCT":     "fg3_pct",
            "FT_PCT":      "ft_pct",
            "REB":         "reb_per_game",
            "AST":         "ast_per_game",
            "TOV":         "tov_per_game",
            "STL":         "stl_per_game",
            "BLK":         "blk_per_game",
            "OREB":        "oreb_per_game",
            "DREB":        "dreb_per_game",
            "PLUS_MINUS":  "plus_minus",
            "OFF_RATING":  "off_rating",
            "DEF_RATING":  "def_rating",
            "NET_RATING":  "net_rating",
            "EFG_PCT":     "efg_pct",
            "TS_PCT":      "ts_pct",
            "PACE":        "pace",
            "PIE":         "pie",
            "TM_TOV_PCT":  "tov_pct",
            "OREB_PCT":    "oreb_pct",
            "DREB_PCT":    "dreb_pct",
        })

        df["win_pct"] = df["wins"] / df["games_played"]
        df["season"]  = season

        # Simple Rating System approximation
        df["srs"] = df["net_rating"] * 0.95

        df = df.reset_index(drop=True)
        _save_cache(cache_key, df)
        return df

    except Exception as e:
        logger.error(f"Error fetching team stats: {e}")
        return _mock_team_stats()


# ─── Player Stats ─────────────────────────────────────────────────────────────

def get_player_stats(
    season: str = CURRENT_SEASON,
    season_type: str = SEASON_TYPE_REGULAR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch player-level statistics: points, rebounds, assists, efficiency,
    usage rate, True Shooting %, PER approximation, minutes.
    """
    cache_key = f"player_stats_{season}_{season_type.replace(' ', '_')}"
    if not force_refresh:
        cached = _load_cache(cache_key, ttl=3600)
        if cached is not None:
            return cached

    if not NBA_API_AVAILABLE:
        return _mock_player_stats()

    try:
        time.sleep(API_SLEEP)
        base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Base",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]

        time.sleep(API_SLEEP)
        advanced = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]

        df = base.merge(
            advanced[["PLAYER_ID", "USG_PCT", "OFF_RATING", "DEF_RATING",
                       "NET_RATING", "AST_PCT", "AST_TO", "TS_PCT", "EFG_PCT",
                       "PIE"]],
            on="PLAYER_ID", how="left"
        )

        df = df.rename(columns={
            "PLAYER_NAME": "player_name",
            "PLAYER_ID":   "player_id",
            "TEAM_ABBREVIATION": "team_abbr",
            "TEAM_ID":     "team_id",
            "GP":          "games_played",
            "MIN":         "min_per_game",
            "PTS":         "pts_per_game",
            "REB":         "reb_per_game",
            "AST":         "ast_per_game",
            "STL":         "stl_per_game",
            "BLK":         "blk_per_game",
            "TOV":         "tov_per_game",
            "FG_PCT":      "fg_pct",
            "FG3_PCT":     "fg3_pct",
            "FT_PCT":      "ft_pct",
            "PLUS_MINUS":  "plus_minus",
            "USG_PCT":     "usg_pct",
            "TS_PCT":      "ts_pct",
            "EFG_PCT":     "efg_pct",
            "OFF_RATING":  "off_rating",
            "DEF_RATING":  "def_rating",
            "NET_RATING":  "net_rating",
            "PIE":         "pie",
        })

        # Approximate PER (simplified)
        df["per_approx"] = (
            df["pts_per_game"] + df["reb_per_game"] * 1.2 +
            df["ast_per_game"] * 1.5 + df["stl_per_game"] * 2 +
            df["blk_per_game"] * 2 - df["tov_per_game"] * 1.5
        ) / df["min_per_game"].clip(lower=1)

        # Star score (0-100)
        df["star_score"] = (
            df["pts_per_game"] * 0.5 +
            df["reb_per_game"] * 0.3 +
            df["ast_per_game"] * 0.4 +
            df.get("usg_pct", 0) * 10
        ).clip(upper=100)

        df["season"] = season
        df = df.reset_index(drop=True)
        _save_cache(cache_key, df)
        return df

    except Exception as e:
        logger.error(f"Error fetching player stats: {e}")
        return _mock_player_stats()


# ─── Team Game Logs ───────────────────────────────────────────────────────────

def get_team_game_logs(
    team_id: int,
    season: str = CURRENT_SEASON,
    season_type: str = SEASON_TYPE_REGULAR,
) -> pd.DataFrame:
    """Fetch game-by-game log for a specific team."""
    cache_key = f"gamelogs_{team_id}_{season}_{season_type.replace(' ', '_')}"
    cached = _load_cache(cache_key, ttl=1800)
    if cached is not None:
        return cached

    if not NBA_API_AVAILABLE:
        return pd.DataFrame()

    try:
        time.sleep(API_SLEEP)
        logs = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable=season_type,
        ).get_data_frames()[0]

        logs = logs.rename(columns={
            "TEAM_ID":    "team_id",
            "GAME_ID":    "game_id",
            "GAME_DATE":  "game_date",
            "MATCHUP":    "matchup",
            "WL":         "wl",
            "PTS":        "pts",
            "FG_PCT":     "fg_pct",
            "FG3_PCT":    "fg3_pct",
            "FT_PCT":     "ft_pct",
            "REB":        "reb",
            "AST":        "ast",
            "STL":        "stl",
            "BLK":        "blk",
            "TOV":        "tov",
            "PLUS_MINUS": "plus_minus",
        })
        logs["win"] = (logs["wl"] == "W").astype(int)
        logs["game_date"] = pd.to_datetime(logs["game_date"])
        logs = logs.sort_values("game_date", ascending=False).reset_index(drop=True)

        _save_cache(cache_key, logs)
        return logs
    except Exception as e:
        logger.error(f"Error fetching game logs for team {team_id}: {e}")
        return pd.DataFrame()


# ─── Recent Form ──────────────────────────────────────────────────────────────

def get_recent_form(team_id: int, n_games: int = 10) -> Dict[str, float]:
    """
    Calculate recent form metrics from last N games.
    Returns: win_pct, avg_pts, avg_pts_allowed, avg_margin.
    """
    logs = get_team_game_logs(team_id)
    if logs.empty or len(logs) < 3:
        return {"last_n_win_pct": 0.5, "last_n_avg_pts": 110.0,
                "last_n_avg_margin": 0.0, "streak": 0}

    recent = logs.head(n_games)
    win_pct    = recent["win"].mean()
    avg_pts    = recent["pts"].mean()
    avg_margin = recent["plus_minus"].mean()

    # Streak (positive = win streak, negative = loss streak)
    streak = 0
    last_result = recent.iloc[0]["win"]
    for _, row in recent.iterrows():
        if row["win"] == last_result:
            streak += (1 if last_result else -1)
        else:
            break

    return {
        "last_n_win_pct":   float(win_pct),
        "last_n_avg_pts":   float(avg_pts),
        "last_n_avg_margin":float(avg_margin),
        "streak":           int(streak),
    }


# ─── Head-to-Head ─────────────────────────────────────────────────────────────

def get_head_to_head(
    team_abbr_home: str,
    team_abbr_away: str,
    n_seasons: int = 3,
) -> Dict[str, float]:
    """
    Fetch head-to-head record and average margin between two teams
    over the last N regular seasons.
    """
    cache_key = f"h2h_{team_abbr_home}_{team_abbr_away}_{n_seasons}"
    cached = _load_cache(cache_key, ttl=86400)
    if cached is not None:
        return cached

    if not NBA_API_AVAILABLE:
        return {"h2h_win_pct": 0.5, "h2h_avg_margin": 0.0, "h2h_games": 0}

    try:
        time.sleep(API_SLEEP)
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=_get_team_id(team_abbr_home),
            vs_team_id_nullable=_get_team_id(team_abbr_away),
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]

        if finder.empty:
            result = {"h2h_win_pct": 0.5, "h2h_avg_margin": 0.0, "h2h_games": 0}
        else:
            finder["WIN"] = (finder["WL"] == "W").astype(int)
            result = {
                "h2h_win_pct":    float(finder["WIN"].mean()),
                "h2h_avg_margin": float(finder["PLUS_MINUS"].mean()),
                "h2h_games":      int(len(finder)),
            }

        _save_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error fetching H2H stats: {e}")
        return {"h2h_win_pct": 0.5, "h2h_avg_margin": 0.0, "h2h_games": 0}


# ─── Live Scoreboard ──────────────────────────────────────────────────────────

def get_live_scores() -> List[Dict]:
    """
    Fetch today's live / most recent game scores.
    Returns a list of game dicts with scores, status, period.
    """
    cache_key = "live_scores"
    cached = _load_cache(cache_key, ttl=30)  # very short TTL for live data
    if cached is not None:
        return cached

    games = []
    try:
        if not NBA_API_AVAILABLE:
            return []

        today = datetime.now().strftime("%m/%d/%Y")
        time.sleep(API_SLEEP)
        board = scoreboardv2.ScoreboardV2(
            game_date=today,
            league_id="00",
        ).get_data_frames()

        line_score = board[1]  # LineScore
        header     = board[0]  # GameHeader

        for _, game in header.iterrows():
            game_id = game["GAME_ID"]
            teams   = line_score[line_score["GAME_ID"] == game_id]
            if len(teams) < 2:
                continue
            away_team = teams.iloc[0]
            home_team = teams.iloc[1]
            games.append({
                "game_id":       game_id,
                "status":        game.get("GAME_STATUS_TEXT", ""),
                "period":        int(game.get("LIVE_PERIOD", 0)),
                "home_team":     home_team.get("TEAM_ABBREVIATION", ""),
                "away_team":     away_team.get("TEAM_ABBREVIATION", ""),
                "home_score":    int(home_team.get("PTS", 0) or 0),
                "away_score":    int(away_team.get("PTS", 0) or 0),
                "game_date":     str(game.get("GAME_DATE_EST", "")),
            })

        _save_cache(cache_key, games)
        return games
    except Exception as e:
        logger.error(f"Error fetching live scores: {e}")
        return []


# ─── Standings ────────────────────────────────────────────────────────────────

def get_standings(season: str = CURRENT_SEASON, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch current league standings."""
    cache_key = f"standings_{season}"
    if not force_refresh:
        cached = _load_cache(cache_key, ttl=3600)
        if cached is not None:
            return cached

    if not NBA_API_AVAILABLE:
        return pd.DataFrame()

    try:
        time.sleep(API_SLEEP)
        df = leaguestandings.LeagueStandings(
            season=season,
            season_type="Regular Season",
        ).get_data_frames()[0]

        df = df.rename(columns={
            "TeamName":      "team_name",
            "TeamID":        "team_id",
            "TeamCity":      "team_city",
            "Conference":    "conference",
            "ConferenceRecord": "conf_record",
            "WINS":          "wins",
            "LOSSES":        "losses",
            "WinPCT":        "win_pct",
            "HOME":          "home_record",
            "ROAD":          "away_record",
            "L10":           "last10",
            "CurrentStreak": "streak",
            "PointsPG":      "pts_per_game",
            "OppPointsPG":   "opp_pts_per_game",
        })

        _save_cache(cache_key, df)
        return df
    except Exception as e:
        logger.error(f"Error fetching standings: {e}")
        return pd.DataFrame()


# ─── Historical Games (for training) ─────────────────────────────────────────

def get_historical_games(seasons: List[str] = None) -> pd.DataFrame:
    """
    Fetch all games across multiple seasons for model training.
    Includes box scores and outcomes.
    """
    seasons = seasons or TRAINING_SEASONS
    cache_key = f"historical_games_{'_'.join(seasons[-2:])}"
    cached = _load_cache(cache_key, ttl=86400 * 7)  # 1 week TTL
    if cached is not None:
        return cached

    if not NBA_API_AVAILABLE:
        return _mock_historical_games()

    all_games = []
    for season in seasons:
        for season_type in [SEASON_TYPE_REGULAR, SEASON_TYPE_PLAYOFFS]:
            try:
                time.sleep(API_SLEEP)
                games = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable=season_type,
                    league_id_nullable="00",
                ).get_data_frames()[0]
                games["season_type"] = season_type
                games["season"]      = season
                all_games.append(games)
                logger.info(f"Fetched {len(games)} games for {season} {season_type}")
            except Exception as e:
                logger.error(f"Error fetching games for {season} {season_type}: {e}")

    if not all_games:
        return _mock_historical_games()

    df = pd.concat(all_games, ignore_index=True)
    df["WIN"] = (df["WL"] == "W").astype(int)
    _save_cache(cache_key, df)
    return df


# ─── Helper: Get Team ID from Abbreviation ───────────────────────────────────

_TEAM_ID_CACHE: Dict[str, int] = {}

def _get_team_id(abbr: str) -> Optional[int]:
    if abbr in _TEAM_ID_CACHE:
        return _TEAM_ID_CACHE[abbr]
    if not NBA_API_AVAILABLE:
        return None
    try:
        all_teams = nba_teams_static.get_teams()
        for t in all_teams:
            if t["abbreviation"] == abbr:
                _TEAM_ID_CACHE[abbr] = t["id"]
                return t["id"]
    except Exception:
        pass
    return None


def get_all_nba_teams() -> List[Dict]:
    """Return list of all NBA teams with id, name, abbreviation."""
    if not NBA_API_AVAILABLE:
        return []
    return nba_teams_static.get_teams()


# ─── Mock Data (fallback when nba_api unavailable) ───────────────────────────

def _mock_team_stats() -> pd.DataFrame:
    """Return synthetic team stats for development/testing."""
    import random
    teams = [
        ("Boston Celtics", "BOS"), ("Denver Nuggets", "DEN"),
        ("Oklahoma City Thunder", "OKC"), ("Cleveland Cavaliers", "CLE"),
        ("New York Knicks", "NYK"), ("Indiana Pacers", "IND"),
        ("Milwaukee Bucks", "MIL"), ("Orlando Magic", "ORL"),
        ("Minnesota Timberwolves", "MIN"), ("Los Angeles Lakers", "LAL"),
        ("Golden State Warriors", "GSW"), ("Memphis Grizzlies", "MEM"),
        ("Dallas Mavericks", "DAL"), ("Los Angeles Clippers", "LAC"),
        ("Phoenix Suns", "PHX"), ("Sacramento Kings", "SAC"),
    ]
    rows = []
    for i, (name, abbr) in enumerate(teams):
        rng = random.Random(i)
        rows.append({
            "team_id":       1610612737 + i,
            "team_name":     name,
            "team_abbr":     abbr,
            "games_played":  82,
            "wins":          rng.randint(28, 64),
            "losses":        82 - rng.randint(28, 64),
            "pts_per_game":  rng.uniform(105, 122),
            "fg_pct":        rng.uniform(0.44, 0.52),
            "fg3_pct":       rng.uniform(0.33, 0.42),
            "ft_pct":        rng.uniform(0.74, 0.84),
            "reb_per_game":  rng.uniform(41, 48),
            "ast_per_game":  rng.uniform(23, 30),
            "stl_per_game":  rng.uniform(5.5, 9.0),
            "blk_per_game":  rng.uniform(3.5, 6.5),
            "tov_per_game":  rng.uniform(11, 16),
            "oreb_per_game": rng.uniform(9, 13),
            "dreb_per_game": rng.uniform(30, 36),
            "off_rating":    rng.uniform(108, 122),
            "def_rating":    rng.uniform(106, 120),
            "net_rating":    rng.uniform(-5, 12),
            "efg_pct":       rng.uniform(0.51, 0.60),
            "ts_pct":        rng.uniform(0.55, 0.63),
            "pace":          rng.uniform(96, 102),
            "pie":           rng.uniform(0.48, 0.56),
            "tov_pct":       rng.uniform(11, 16),
            "oreb_pct":      rng.uniform(22, 32),
            "dreb_pct":      rng.uniform(68, 78),
            "srs":           rng.uniform(-5, 10),
            "win_pct":       rng.uniform(0.35, 0.78),
            "opp_fg_pct":    rng.uniform(0.44, 0.50),
            "opp_fg3_pct":   rng.uniform(0.33, 0.39),
            "plus_minus":    rng.uniform(-4, 10),
            "season":        CURRENT_SEASON,
        })
    return pd.DataFrame(rows)


def _mock_player_stats() -> pd.DataFrame:
    import random
    players = [
        ("Jayson Tatum", "BOS"), ("Nikola Jokic", "DEN"),
        ("Shai Gilgeous-Alexander", "OKC"), ("Donovan Mitchell", "CLE"),
        ("Karl-Anthony Towns", "NYK"), ("Tyrese Haliburton", "IND"),
        ("Giannis Antetokounmpo", "MIL"), ("Paolo Banchero", "ORL"),
        ("Anthony Edwards", "MIN"), ("LeBron James", "LAL"),
        ("Stephen Curry", "GSW"), ("Ja Morant", "MEM"),
        ("Luka Doncic", "DAL"), ("Kawhi Leonard", "LAC"),
        ("Kevin Durant", "PHX"), ("De'Aaron Fox", "SAC"),
    ]
    rows = []
    for i, (name, team) in enumerate(players):
        rng = random.Random(i + 100)
        rows.append({
            "player_id":    203507 + i,
            "player_name":  name,
            "team_abbr":    team,
            "team_id":      1610612737 + i,
            "games_played": rng.randint(55, 78),
            "min_per_game": rng.uniform(30, 37),
            "pts_per_game": rng.uniform(18, 35),
            "reb_per_game": rng.uniform(4, 12),
            "ast_per_game": rng.uniform(3, 10),
            "stl_per_game": rng.uniform(0.8, 1.8),
            "blk_per_game": rng.uniform(0.4, 2.0),
            "tov_per_game": rng.uniform(2, 5),
            "fg_pct":       rng.uniform(0.44, 0.57),
            "fg3_pct":      rng.uniform(0.33, 0.44),
            "ft_pct":       rng.uniform(0.72, 0.90),
            "plus_minus":   rng.uniform(-2, 10),
            "usg_pct":      rng.uniform(0.22, 0.36),
            "ts_pct":       rng.uniform(0.55, 0.68),
            "efg_pct":      rng.uniform(0.50, 0.62),
            "off_rating":   rng.uniform(108, 125),
            "def_rating":   rng.uniform(106, 120),
            "net_rating":   rng.uniform(-2, 14),
            "pie":          rng.uniform(0.12, 0.22),
            "per_approx":   rng.uniform(0.8, 2.2),
            "star_score":   rng.uniform(30, 80),
            "season":       CURRENT_SEASON,
        })
    return pd.DataFrame(rows)


def _mock_historical_games() -> pd.DataFrame:
    """Synthetic game results for testing model training."""
    import random
    rows = []
    team_ids = list(range(1610612737, 1610612767))
    for i in range(2000):
        rng = random.Random(i)
        t1, t2 = rng.sample(team_ids, 2)
        pts = rng.randint(88, 135)
        opp = rng.randint(88, 135)
        rows.append({
            "TEAM_ID":       t1,
            "OPPONENT_TEAM_ID": t2,
            "WL":            "W" if pts > opp else "L",
            "WIN":           1 if pts > opp else 0,
            "PTS":           pts,
            "OPP_PTS":       opp,
            "PLUS_MINUS":    pts - opp,
            "GAME_DATE":     f"2024-0{rng.randint(1,9)}-{rng.randint(10,28)}",
            "season":        CURRENT_SEASON,
            "season_type":   SEASON_TYPE_REGULAR,
        })
    return pd.DataFrame(rows)
