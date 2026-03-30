"""
Odds Scraper Module
Fetches betting odds, public money percentages, and line movement data
from Odds Shark and Action Network to enrich ML features.

Data pulled:
  - Point spreads, moneylines, over/unders
  - Public betting % (bets and money) from Action Network
  - Line movement (opening vs current spread)
  - Against-The-Spread (ATS) team records from Odds Shark
"""

import re
import time
import json
import logging
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from config import DATA_DIR, CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

# ─── HTTP Headers (rotate to avoid blocks) ───────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ─── Action Network API (undocumented but public JSON endpoint) ───────────────
ACTION_NETWORK_API = "https://api.actionnetwork.com/web/v1/games"
ACTION_NETWORK_ODDS = "https://api.actionnetwork.com/web/v1/scoreboard/nba"

# ─── Odds Shark URLs ──────────────────────────────────────────────────────────
ODDS_SHARK_ODDS  = "https://www.oddsshark.com/nba/odds"
ODDS_SHARK_ATS   = "https://www.oddsshark.com/nba/ats-standings"
ODDS_SHARK_POWER = "https://www.oddsshark.com/nba/power-rankings"

# Team name normalization maps
ODDSSHARK_NAME_MAP = {
    "76ers": "PHI", "Bucks": "MIL", "Bulls": "CHI", "Cavaliers": "CLE",
    "Celtics": "BOS", "Clippers": "LAC", "Grizzlies": "MEM", "Hawks": "ATL",
    "Heat": "MIA", "Hornets": "CHA", "Jazz": "UTA", "Kings": "SAC",
    "Knicks": "NYK", "Lakers": "LAL", "Magic": "ORL", "Mavericks": "DAL",
    "Nets": "BKN", "Nuggets": "DEN", "Pacers": "IND", "Pelicans": "NOP",
    "Pistons": "DET", "Raptors": "TOR", "Rockets": "HOU", "Spurs": "SAS",
    "Suns": "PHX", "Thunder": "OKC", "Timberwolves": "MIN", "Trail Blazers": "POR",
    "Warriors": "GSW", "Wizards": "WAS",
}

ACTION_NAME_MAP = {
    "Philadelphia 76ers": "PHI", "Milwaukee Bucks": "MIL", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Boston Celtics": "BOS", "LA Clippers": "LAC",
    "Memphis Grizzlies": "MEM", "Atlanta Hawks": "ATL", "Miami Heat": "MIA",
    "Charlotte Hornets": "CHA", "Utah Jazz": "UTA", "Sacramento Kings": "SAC",
    "New York Knicks": "NYK", "Los Angeles Lakers": "LAL", "Orlando Magic": "ORL",
    "Dallas Mavericks": "DAL", "Brooklyn Nets": "BKN", "Denver Nuggets": "DEN",
    "Indiana Pacers": "IND", "New Orleans Pelicans": "NOP", "Detroit Pistons": "DET",
    "Toronto Raptors": "TOR", "Houston Rockets": "HOU", "San Antonio Spurs": "SAS",
    "Phoenix Suns": "PHX", "Oklahoma City Thunder": "OKC",
    "Minnesota Timberwolves": "MIN", "Portland Trail Blazers": "POR",
    "Golden State Warriors": "GSW", "Washington Wizards": "WAS",
}


# ─── Cache Helpers ────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return DATA_DIR / f"odds_{key}.pkl"


def _load_cache(key: str, ttl: int = CACHE_TTL_SECONDS) -> Optional[object]:
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
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.warning(f"Cache write failed for {key}: {e}")


# ─── Moneyline ↔ Probability Utilities ───────────────────────────────────────

def moneyline_to_implied_prob(ml: float) -> float:
    """Convert American odds moneyline to implied win probability (0-1)."""
    if ml is None or np.isnan(ml):
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def implied_prob_to_moneyline(prob: float) -> float:
    """Convert implied probability to American moneyline."""
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        return -(prob / (1 - prob)) * 100
    else:
        return ((1 - prob) / prob) * 100


def remove_vig(home_prob: float, away_prob: float) -> Tuple[float, float]:
    """Remove bookmaker vig to get true implied probabilities."""
    total = home_prob + away_prob
    if total == 0:
        return 0.5, 0.5
    return home_prob / total, away_prob / total


# ─── Action Network Scraper ───────────────────────────────────────────────────

def get_action_network_odds(force_refresh: bool = False) -> Dict:
    """
    Fetch current NBA game odds and public betting percentages from Action Network.

    Returns dict keyed by (away_abbr, home_abbr) tuples with:
      - spread, total, home_ml, away_ml
      - public_bet_pct_home, public_money_pct_home
      - line_movement (opening spread - current spread)
      - implied_win_prob_home
    """
    cache_key = "action_network_odds"
    if not force_refresh:
        cached = _load_cache(cache_key, ttl=180)
        if cached is not None:
            return cached

    today_str = date.today().strftime("%Y-%m-%d")
    url = f"{ACTION_NETWORK_API}?sport=nba&date={today_str}&bookIds=15,30,76,123&include=odds,teams"

    try:
        resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = _parse_action_network(data)
        _save_cache(cache_key, result)
        logger.info(f"Action Network: fetched {len(result)} games")
        return result
    except Exception as e:
        logger.warning(f"Action Network primary API failed: {e}")

    # Fallback: try scoreboard endpoint
    try:
        resp = SESSION.get(ACTION_NETWORK_ODDS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = _parse_action_network(data)
        _save_cache(cache_key, result)
        return result
    except Exception as e2:
        logger.warning(f"Action Network fallback failed: {e2}")

    # Return mock data during off-season / scrape failures
    return _mock_action_network_odds()


def _parse_action_network(data: Dict) -> Dict:
    """Parse Action Network JSON response into our standard format."""
    result = {}
    games = data.get("games", [])

    for game in games:
        try:
            teams = game.get("teams", {})
            home_team = teams.get("home", {}).get("full_name", "")
            away_team = teams.get("away", {}).get("full_name", "")

            home_abbr = ACTION_NAME_MAP.get(home_team)
            away_abbr = ACTION_NAME_MAP.get(away_team)
            if not home_abbr or not away_abbr:
                continue

            # Odds aggregated across books
            odds = game.get("odds", [{}])
            best_odds = odds[0] if odds else {}

            spread = float(best_odds.get("spread", 0) or 0)
            total  = float(best_odds.get("total",  220) or 220)
            home_ml = float(best_odds.get("ml_home", -110) or -110)
            away_ml = float(best_odds.get("ml_away", -110) or -110)

            # Opening line
            open_spread = float(best_odds.get("open_spread", spread) or spread)
            line_movement = round(open_spread - spread, 1)

            # Public betting %
            consensus = game.get("consensus", {})
            home_bet_pct   = float(consensus.get("spread_home_pct", 50) or 50)
            away_bet_pct   = 100.0 - home_bet_pct
            home_money_pct = float(consensus.get("spread_home_money_pct", 50) or 50)
            away_money_pct = 100.0 - home_money_pct

            home_impl  = moneyline_to_implied_prob(home_ml)
            away_impl  = moneyline_to_implied_prob(away_ml)
            home_true, away_true = remove_vig(home_impl, away_impl)

            key = (away_abbr, home_abbr)
            result[key] = {
                "home_abbr":            home_abbr,
                "away_abbr":            away_abbr,
                "spread":               spread,        # negative = home favored
                "total":                total,
                "home_ml":              home_ml,
                "away_ml":              away_ml,
                "line_movement":        line_movement, # positive = moved toward home
                "public_bet_pct_home":  home_bet_pct,
                "public_bet_pct_away":  away_bet_pct,
                "public_money_pct_home": home_money_pct,
                "public_money_pct_away": away_money_pct,
                "implied_win_prob_home": round(home_true, 4),
                "implied_win_prob_away": round(away_true, 4),
                "sharp_money_indicator": home_money_pct - home_bet_pct,  # + = sharp on home
                "game_time":            game.get("start_time", ""),
                "source": "action_network",
            }
        except Exception as e:
            logger.debug(f"Skipping game parse: {e}")
            continue

    return result


# ─── Odds Shark Scraper ───────────────────────────────────────────────────────

def get_odds_shark_odds(force_refresh: bool = False) -> Dict:
    """
    Scrape current NBA game odds from Odds Shark.
    Returns dict keyed by (away_abbr, home_abbr) with spread, total, moneylines.
    """
    cache_key = "odds_shark_odds"
    if not force_refresh:
        cached = _load_cache(cache_key, ttl=300)
        if cached is not None:
            return cached

    try:
        resp = SESSION.get(ODDS_SHARK_ODDS, timeout=12)
        resp.raise_for_status()
        result = _parse_odds_shark_html(resp.text)
        if result:
            _save_cache(cache_key, result)
            logger.info(f"Odds Shark: fetched {len(result)} games")
            return result
    except Exception as e:
        logger.warning(f"Odds Shark HTML scrape failed: {e}")

    return _mock_odds_shark_odds()


def _parse_odds_shark_html(html: str) -> Dict:
    """Parse Odds Shark HTML to extract game odds tables."""
    result = {}
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Odds Shark embeds data in JSON within a script tag
        for script in soup.find_all("script"):
            text = script.string or ""
            if "matchups" in text and "spread" in text:
                # Try to extract embedded JSON
                match = re.search(r'window\.__data\s*=\s*({.*?});', text, re.DOTALL)
                if match:
                    try:
                        raw = json.loads(match.group(1))
                        return _parse_odds_shark_json(raw)
                    except json.JSONDecodeError:
                        pass

        # Fallback: parse HTML tables directly
        tables = soup.find_all("table", class_=re.compile("(?i)odds|game|matchup"))
        for table in tables:
            rows = table.find_all("tr")
            for i, row in enumerate(rows):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 4:
                    try:
                        away_name = cells[0].get_text(strip=True)
                        home_name = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                        spread_txt = cells[2].get_text(strip=True) if len(cells) > 2 else "0"
                        total_txt  = cells[3].get_text(strip=True) if len(cells) > 3 else "220"
                        ml_txt     = cells[4].get_text(strip=True) if len(cells) > 4 else "-110"

                        away_abbr = _normalize_team_name(away_name)
                        home_abbr = _normalize_team_name(home_name)
                        if not away_abbr or not home_abbr:
                            continue

                        spread = float(re.sub(r"[^0-9.\-]", "", spread_txt) or 0)
                        total  = float(re.sub(r"[^0-9.]", "", total_txt)    or 220)
                        ml     = float(re.sub(r"[^0-9.\-]", "", ml_txt)     or -110)

                        home_impl = moneyline_to_implied_prob(ml)
                        key = (away_abbr, home_abbr)
                        result[key] = {
                            "spread": spread,
                            "total":  total,
                            "home_ml": ml,
                            "implied_win_prob_home": home_impl,
                            "source": "odds_shark_table",
                        }
                    except Exception:
                        continue
    except Exception as e:
        logger.warning(f"Odds Shark parse error: {e}")

    return result


def _parse_odds_shark_json(data: Dict) -> Dict:
    """Parse Odds Shark embedded JSON data."""
    result = {}
    try:
        matchups = data.get("matchups", [])
        for game in matchups:
            away = game.get("awayTeam", {})
            home = game.get("homeTeam", {})
            away_name = away.get("shortName", away.get("name", ""))
            home_name = home.get("shortName", home.get("name", ""))

            away_abbr = _normalize_team_name(away_name)
            home_abbr = _normalize_team_name(home_name)
            if not away_abbr or not home_abbr:
                continue

            odds = game.get("odds", {})
            spread = float(odds.get("spreadHome", 0) or 0)
            total  = float(odds.get("total",     220) or 220)
            home_ml = float(odds.get("moneylineHome", -110) or -110)
            away_ml = float(odds.get("moneylineAway", -110) or -110)

            home_impl = moneyline_to_implied_prob(home_ml)
            away_impl = moneyline_to_implied_prob(away_ml)
            home_true, away_true = remove_vig(home_impl, away_impl)

            key = (away_abbr, home_abbr)
            result[key] = {
                "spread": spread,
                "total":  total,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "implied_win_prob_home": round(home_true, 4),
                "implied_win_prob_away": round(away_true, 4),
                "source": "odds_shark_json",
            }
    except Exception as e:
        logger.warning(f"Odds Shark JSON parse error: {e}")
    return result


# ─── ATS Record Scraper (Odds Shark) ─────────────────────────────────────────

def get_ats_records(force_refresh: bool = False) -> Dict:
    """
    Scrape Against-The-Spread (ATS) season records for all teams from Odds Shark.
    Returns dict: {team_abbr: {"ats_wins": int, "ats_losses": int, "ats_pushes": int,
                                "ats_pct": float, "ou_over_pct": float}}
    """
    cache_key = "ats_records"
    if not force_refresh:
        cached = _load_cache(cache_key, ttl=3600)  # refresh hourly
        if cached is not None:
            return cached

    try:
        resp = SESSION.get(ODDS_SHARK_ATS, timeout=12)
        resp.raise_for_status()
        result = _parse_ats_table(resp.text)
        if result:
            _save_cache(cache_key, result)
            logger.info(f"ATS records: {len(result)} teams")
            return result
    except Exception as e:
        logger.warning(f"ATS scrape failed: {e}")

    return _mock_ats_records()


def _parse_ats_table(html: str) -> Dict:
    """Parse Odds Shark ATS standings table."""
    result = {}
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return result
        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            team_name = cells[0].get_text(strip=True)
            team_abbr = _normalize_team_name(team_name)
            if not team_abbr:
                continue

            ats_txt = cells[1].get_text(strip=True)   # e.g. "28-22-3"
            ou_txt  = cells[2].get_text(strip=True) if len(cells) > 2 else "25-25"

            ats_parts = re.findall(r"\d+", ats_txt)
            ou_parts  = re.findall(r"\d+", ou_txt)

            ats_w = int(ats_parts[0]) if len(ats_parts) > 0 else 0
            ats_l = int(ats_parts[1]) if len(ats_parts) > 1 else 0
            ats_p = int(ats_parts[2]) if len(ats_parts) > 2 else 0
            total = ats_w + ats_l + ats_p
            ats_pct = ats_w / total if total > 0 else 0.5

            ou_over  = int(ou_parts[0]) if len(ou_parts) > 0 else 0
            ou_under = int(ou_parts[1]) if len(ou_parts) > 1 else 0
            ou_total = ou_over + ou_under
            ou_over_pct = ou_over / ou_total if ou_total > 0 else 0.5

            result[team_abbr] = {
                "ats_wins":    ats_w,
                "ats_losses":  ats_l,
                "ats_pushes":  ats_p,
                "ats_pct":     round(ats_pct, 4),
                "ou_over_pct": round(ou_over_pct, 4),
            }
    except Exception as e:
        logger.warning(f"ATS parse error: {e}")
    return result


# ─── Combined Odds Lookup ─────────────────────────────────────────────────────

def get_game_odds(home_team_abbr: str, away_team_abbr: str) -> Dict:
    """
    Get the best available odds data for a specific matchup.
    Merges Action Network (public %) with Odds Shark (ATS, backup lines).

    Returns a unified odds dict with all available fields.
    """
    key = (away_team_abbr.upper(), home_team_abbr.upper())
    rev_key = (home_team_abbr.upper(), away_team_abbr.upper())

    # Try Action Network first (has public % data)
    an_odds = get_action_network_odds()
    game = an_odds.get(key) or an_odds.get(rev_key, {})

    # Supplement with Odds Shark
    os_odds = get_odds_shark_odds()
    os_game = os_odds.get(key) or os_odds.get(rev_key, {})

    # Merge: Action Network base + fill gaps from Odds Shark
    merged = {**os_game, **game}

    # ATS record for both teams
    ats = get_ats_records()
    merged["home_ats_pct"]    = ats.get(home_team_abbr, {}).get("ats_pct", 0.5)
    merged["away_ats_pct"]    = ats.get(away_team_abbr, {}).get("ats_pct", 0.5)
    merged["home_ou_over_pct"] = ats.get(home_team_abbr, {}).get("ou_over_pct", 0.5)
    merged["away_ou_over_pct"] = ats.get(away_team_abbr, {}).get("ou_over_pct", 0.5)

    # Ensure defaults
    defaults = {
        "spread":                 0.0,
        "total":                  220.0,
        "home_ml":                -110.0,
        "away_ml":                -110.0,
        "implied_win_prob_home":  0.5,
        "implied_win_prob_away":  0.5,
        "line_movement":          0.0,
        "public_bet_pct_home":    50.0,
        "public_money_pct_home":  50.0,
        "sharp_money_indicator":  0.0,
    }
    for k, v in defaults.items():
        merged.setdefault(k, v)

    return merged


def get_all_current_odds() -> Dict:
    """
    Get odds for all games currently available.
    Returns merged dict from both sources.
    """
    an_data = get_action_network_odds()
    os_data = get_odds_shark_odds()

    # Merge — Action Network takes priority
    combined = {**os_data, **an_data}

    # Append ATS records inline
    ats = get_ats_records()
    for key, game in combined.items():
        home = game.get("home_abbr", key[1] if len(key) > 1 else "")
        away = game.get("away_abbr", key[0] if len(key) > 0 else "")
        game["home_ats_pct"] = ats.get(home, {}).get("ats_pct", 0.5)
        game["away_ats_pct"] = ats.get(away, {}).get("ats_pct", 0.5)

    return combined


def get_odds_features_for_team(
    team_abbr: str,
    opponent_abbr: str,
    is_home: bool = True
) -> Dict[str, float]:
    """
    Return odds-based feature dict for a single team in a matchup.
    Used directly in the feature engineering pipeline.
    """
    home_abbr = team_abbr if is_home else opponent_abbr
    away_abbr = opponent_abbr if is_home else team_abbr

    odds = get_game_odds(home_abbr, away_abbr)
    ats  = get_ats_records()

    if is_home:
        implied_prob     = odds.get("implied_win_prob_home", 0.5)
        opp_implied_prob = odds.get("implied_win_prob_away", 0.5)
        bet_pct          = odds.get("public_bet_pct_home", 50.0)
        money_pct        = odds.get("public_money_pct_home", 50.0)
        spread_val       = odds.get("spread", 0.0)   # negative = home favored
    else:
        implied_prob     = odds.get("implied_win_prob_away", 0.5)
        opp_implied_prob = odds.get("implied_win_prob_home", 0.5)
        bet_pct          = odds.get("public_bet_pct_away", 50.0)
        money_pct        = odds.get("public_money_pct_away", 50.0)
        spread_val       = -odds.get("spread", 0.0)  # flip for away team

    sharp_indicator = money_pct - bet_pct  # + = sharp money on this team

    return {
        "market_win_prob":      implied_prob,
        "opp_market_win_prob":  opp_implied_prob,
        "market_spread":        spread_val,
        "market_total":         odds.get("total", 220.0),
        "public_bet_pct":       bet_pct / 100.0,
        "public_money_pct":     money_pct / 100.0,
        "sharp_money_indicator": sharp_indicator / 100.0,
        "line_movement":        odds.get("line_movement", 0.0),
        "ats_pct":              ats.get(team_abbr, {}).get("ats_pct", 0.5),
        "opp_ats_pct":          ats.get(opponent_abbr, {}).get("ats_pct", 0.5),
        "ou_over_pct":          ats.get(team_abbr, {}).get("ou_over_pct", 0.5),
        # Differential: team vs market consensus
        "model_vs_market_edge": 0.0,  # filled later when ML prob is known
    }


# ─── Helper: Team Name Normalization ─────────────────────────────────────────

def _normalize_team_name(name: str) -> Optional[str]:
    """Fuzzy-match a team name string to an NBA abbreviation."""
    if not name:
        return None
    name = name.strip()

    # Direct match in map
    for key, abbr in ODDSSHARK_NAME_MAP.items():
        if key.lower() in name.lower():
            return abbr

    # Full name match
    for full_name, abbr in ACTION_NAME_MAP.items():
        if any(part.lower() in name.lower() for part in full_name.split()):
            return abbr

    return None


# ─── Mock Data (off-season / scrape failure fallbacks) ───────────────────────

def _mock_action_network_odds() -> Dict:
    """Return plausible mock odds for demonstration when scraping is unavailable."""
    logger.info("Using mock Action Network odds data")
    matchups = [
        ("BOS", "CLE", -220,  190, -6.5, 218.5, 62, 67),
        ("OKC", "MEM", -195,  170, -5.5, 222.0, 58, 63),
        ("DEN", "LAC", -175,  155, -4.5, 220.0, 55, 59),
        ("MIN", "PHX", -160,  140, -3.5, 215.5, 54, 57),
        ("NYK", "MIL", -140,  120, -2.5, 221.0, 52, 55),
        ("IND", "MIA", -130,  110, -1.5, 223.5, 51, 53),
        ("ORL", "DAL", -120,  100, -1.0, 212.0, 50, 51),
        ("DET", "CHI",  100, -120,  1.5, 218.0, 48, 49),
    ]
    result = {}
    for home, away, home_ml, away_ml, spread, total, bet_pct, money_pct in matchups:
        hi = moneyline_to_implied_prob(home_ml)
        ai = moneyline_to_implied_prob(away_ml)
        ht, at = remove_vig(hi, ai)
        key = (away, home)
        result[key] = {
            "home_abbr": home, "away_abbr": away,
            "spread": spread, "total": total,
            "home_ml": home_ml, "away_ml": away_ml,
            "line_movement": round(np.random.uniform(-1.5, 1.5), 1),
            "public_bet_pct_home": bet_pct,
            "public_bet_pct_away": 100 - bet_pct,
            "public_money_pct_home": money_pct,
            "public_money_pct_away": 100 - money_pct,
            "implied_win_prob_home": round(ht, 4),
            "implied_win_prob_away": round(at, 4),
            "sharp_money_indicator": money_pct - bet_pct,
            "game_time": "TBD",
            "source": "mock",
        }
    return result


def _mock_odds_shark_odds() -> Dict:
    logger.info("Using mock Odds Shark odds data")
    return {}


def _mock_ats_records() -> Dict:
    """Mock ATS records near league average."""
    from config import NBA_TEAMS
    result = {}
    rng = np.random.default_rng(42)
    for abbr in NBA_TEAMS:
        ats_pct = round(float(rng.normal(0.500, 0.040)), 4)
        result[abbr] = {
            "ats_wins":    int(41 * ats_pct),
            "ats_losses":  int(41 * (1 - ats_pct)),
            "ats_pushes":  2,
            "ats_pct":     ats_pct,
            "ou_over_pct": round(float(rng.normal(0.500, 0.035)), 4),
        }
    return result


# ─── Utility: Odds Summary DataFrame ─────────────────────────────────────────

def get_odds_dataframe() -> pd.DataFrame:
    """
    Return all current game odds as a clean DataFrame for the dashboard.
    """
    all_odds = get_all_current_odds()
    if not all_odds:
        return pd.DataFrame()

    rows = []
    for (away, home), game in all_odds.items():
        rows.append({
            "Away":                game.get("away_abbr", away),
            "Home":                game.get("home_abbr", home),
            "Spread (Home)":       game.get("spread", 0),
            "Total (O/U)":         game.get("total", 220),
            "Home ML":             game.get("home_ml", -110),
            "Away ML":             game.get("away_ml", -110),
            "Market Win% (Home)":  round(game.get("implied_win_prob_home", 0.5) * 100, 1),
            "Public Bets% (Home)": game.get("public_bet_pct_home", 50),
            "Public $% (Home)":    game.get("public_money_pct_home", 50),
            "Sharp Indicator":     round(game.get("sharp_money_indicator", 0), 1),
            "Line Movement":       game.get("line_movement", 0),
            "Home ATS%":           round(game.get("home_ats_pct", 0.5) * 100, 1),
            "Away ATS%":           round(game.get("away_ats_pct", 0.5) * 100, 1),
            "Source":              game.get("source", "unknown"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Market Win% (Home)", ascending=False)
    return df
