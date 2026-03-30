"""
Injury Tracker Module
Scrapes real-time NBA injury reports from ESPN and calculates
impact scores for each team based on player importance.
"""

import time
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from config import (
    DATA_DIR, ESPN_INJURY_URL,
    INJURY_STATUS_WEIGHTS, INJURY_REFRESH_SECONDS
)

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


# ─── Cache Helpers ────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return DATA_DIR / f"{key}.pkl"


def _load_cache(key: str, ttl: int = INJURY_REFRESH_SECONDS) -> Optional[object]:
    path = _cache_path(key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(key: str, data: object) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump(data, f)


# ─── Injury Scraper ───────────────────────────────────────────────────────────

def get_injury_report(force_refresh: bool = False) -> pd.DataFrame:
    """
    Scrape the current NBA injury report from ESPN.
    Returns a DataFrame with columns:
        player_name, team_name, team_abbr, status, injury_type, updated
    """
    cached = _load_cache("injury_report", ttl=INJURY_REFRESH_SECONDS)
    if cached is not None and not force_refresh:
        return cached

    injuries = []
    try:
        resp = requests.get(ESPN_INJURY_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # ESPN injury table structure
        teams = soup.find_all("div", class_="Table__Title")
        tables = soup.find_all("div", class_="injuries")

        for team_div in soup.find_all("section", class_=lambda c: c and "Injury" in c):
            team_name_tag = team_div.find("span", class_="injuries__teamName")
            if not team_name_tag:
                continue
            team_name = team_name_tag.get_text(strip=True)

            rows = team_div.find_all("tr", class_="Table__TR")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                player_name  = cells[0].get_text(strip=True)
                pos          = cells[1].get_text(strip=True)
                injury_type  = cells[2].get_text(strip=True)
                status       = cells[3].get_text(strip=True)
                injuries.append({
                    "player_name":  player_name,
                    "team_name":    team_name,
                    "position":     pos,
                    "injury_type":  injury_type,
                    "status":       status,
                    "updated":      datetime.now().isoformat(),
                })

        # Fallback: try alternate ESPN structure
        if not injuries:
            injuries = _scrape_espn_alternate(soup)

    except requests.RequestException as e:
        logger.warning(f"Could not reach ESPN: {e}. Using fallback injury data.")
        injuries = _get_fallback_injuries()
    except Exception as e:
        logger.error(f"Error parsing ESPN injury page: {e}")
        injuries = _get_fallback_injuries()

    df = pd.DataFrame(injuries) if injuries else _injury_df_empty()

    # Normalize status
    if not df.empty and "status" in df.columns:
        df["status"] = df["status"].str.strip()
        df["status_weight"] = df["status"].map(
            lambda s: next((v for k, v in INJURY_STATUS_WEIGHTS.items()
                            if k.lower() in s.lower()), 0.2)
        )

    _save_cache("injury_report", df)
    return df


def _scrape_espn_alternate(soup: BeautifulSoup) -> List[Dict]:
    """Alternate scrape for ESPN's table-based layout."""
    injuries = []
    try:
        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            if "Name" not in headers:
                continue
            team_header = table.find_previous("h2")
            team_name   = team_header.get_text(strip=True) if team_header else "Unknown"

            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) >= 4:
                    injuries.append({
                        "player_name":  cells[0],
                        "team_name":    team_name,
                        "position":     cells[1] if len(cells) > 1 else "",
                        "injury_type":  cells[2] if len(cells) > 2 else "",
                        "status":       cells[3] if len(cells) > 3 else "Questionable",
                        "updated":      datetime.now().isoformat(),
                    })
    except Exception as e:
        logger.error(f"Alternate scrape failed: {e}")
    return injuries


def _injury_df_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "player_name", "team_name", "position",
        "injury_type", "status", "updated", "status_weight"
    ])


def _get_fallback_injuries() -> List[Dict]:
    """
    Return a small set of well-known, realistic injury entries
    for use when the live scrape fails.
    """
    return [
        {"player_name": "Kawhi Leonard",  "team_name": "LA Clippers",
         "position": "SF", "injury_type": "Knee",   "status": "Out",
         "updated": datetime.now().isoformat()},
        {"player_name": "Joel Embiid",    "team_name": "Philadelphia 76ers",
         "position": "C",  "injury_type": "Knee",   "status": "Out",
         "updated": datetime.now().isoformat()},
    ]


# ─── Injury Impact Calculator ─────────────────────────────────────────────────

def calculate_injury_impact(
    team_name: str,
    player_stats_df: pd.DataFrame,
    injury_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Calculate how much injuries reduce a team's expected performance.

    Returns:
        impact_score     : 0 = no injuries, 1 = entire roster out
        pts_lost_per_game: estimated points lost due to injuries
        injured_players  : list of injured player names
    """
    if injury_df is None:
        injury_df = get_injury_report()

    if injury_df.empty:
        return {"impact_score": 0.0, "pts_lost_per_game": 0.0,
                "injured_players": [], "severity": "None"}

    # Filter injuries for this team
    team_injuries = injury_df[
        injury_df["team_name"].str.lower().str.contains(
            team_name.lower().replace("los angeles", "la"), na=False
        )
    ]

    if team_injuries.empty:
        return {"impact_score": 0.0, "pts_lost_per_game": 0.0,
                "injured_players": [], "severity": "None"}

    # Look up player stats to weigh by importance
    total_impact  = 0.0
    pts_lost      = 0.0
    injured_names = []

    for _, injury_row in team_injuries.iterrows():
        player_name    = injury_row["player_name"]
        status_weight  = injury_row.get("status_weight", 0.5)

        # Find this player in stats
        player_match = player_stats_df[
            player_stats_df["player_name"].str.lower() == player_name.lower()
        ] if not player_stats_df.empty else pd.DataFrame()

        if not player_match.empty:
            pts      = float(player_match.iloc[0].get("pts_per_game", 10))
            usg      = float(player_match.iloc[0].get("usg_pct", 0.20))
            star_sc  = float(player_match.iloc[0].get("star_score", 30))

            # Impact weighted by usage, points, star score
            player_impact = (
                status_weight * (usg * 2 + pts / 30 + star_sc / 100) / 3
            )
            pts_lost     += pts * status_weight
        else:
            # Unknown player – assume bench contributor
            player_impact = status_weight * 0.1
            pts_lost     += 8 * status_weight

        total_impact  += player_impact
        injured_names.append(f"{player_name} ({injury_row.get('status', '?')})")

    # Normalize: assume max plausible impact is 3 stars being out fully
    impact_score = min(total_impact / 1.5, 1.0)

    # Severity label
    severity = "None"
    if impact_score > 0.6:
        severity = "Critical"
    elif impact_score > 0.35:
        severity = "High"
    elif impact_score > 0.15:
        severity = "Moderate"
    elif impact_score > 0.0:
        severity = "Low"

    return {
        "impact_score":      round(impact_score, 4),
        "pts_lost_per_game": round(pts_lost, 1),
        "injured_players":   injured_names,
        "severity":          severity,
    }


def get_team_injury_summary(force_refresh: bool = False) -> Dict[str, Dict]:
    """
    Return a dict mapping team_name → injury impact dict for all teams.
    Used by the model to incorporate injury data into predictions.
    """
    cache_key = "team_injury_summary"
    cached = _load_cache(cache_key, ttl=INJURY_REFRESH_SECONDS)
    if cached is not None and not force_refresh:
        return cached

    from data.nba_data import get_player_stats
    injury_df    = get_injury_report(force_refresh=force_refresh)
    player_stats = get_player_stats()

    summary = {}
    if not injury_df.empty:
        for team_name in injury_df["team_name"].unique():
            summary[team_name] = calculate_injury_impact(
                team_name, player_stats, injury_df
            )

    _save_cache(cache_key, summary)
    return summary


def watch_for_injury_changes(previous_df: pd.DataFrame) -> List[Dict]:
    """
    Compare the current injury report to a previous snapshot.
    Returns a list of change events (new injuries, status changes, players cleared).
    """
    current_df = get_injury_report(force_refresh=True)
    changes    = []

    if previous_df is None or previous_df.empty:
        return changes

    # Players newly added to injury list
    prev_players = set(previous_df["player_name"].tolist())
    curr_players = set(current_df["player_name"].tolist()) if not current_df.empty else set()

    new_injuries = curr_players - prev_players
    cleared      = prev_players - curr_players

    for name in new_injuries:
        row = current_df[current_df["player_name"] == name].iloc[0]
        changes.append({
            "type":   "new_injury",
            "player": name,
            "team":   row.get("team_name", ""),
            "status": row.get("status", ""),
            "injury": row.get("injury_type", ""),
        })

    for name in cleared:
        row = previous_df[previous_df["player_name"] == name].iloc[0]
        changes.append({
            "type":   "cleared",
            "player": name,
            "team":   row.get("team_name", ""),
            "old_status": row.get("status", ""),
        })

    # Status changes
    if not current_df.empty:
        for _, curr_row in current_df.iterrows():
            name = curr_row["player_name"]
            if name in prev_players:
                prev_match = previous_df[previous_df["player_name"] == name]
                if not prev_match.empty:
                    old_status = prev_match.iloc[0].get("status", "")
                    new_status = curr_row.get("status", "")
                    if old_status != new_status:
                        changes.append({
                            "type":       "status_change",
                            "player":     name,
                            "team":       curr_row.get("team_name", ""),
                            "old_status": old_status,
                            "new_status": new_status,
                        })

    return changes
