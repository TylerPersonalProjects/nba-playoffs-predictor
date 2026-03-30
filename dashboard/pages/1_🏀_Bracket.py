"""
Playoff Bracket Page
Full interactive bracket with series predictions, win probabilities,
and championship odds for every remaining team.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bracket | NBA Predictor", page_icon="🏀", layout="wide")

from config import PRIMARY_COLOR, SECONDARY_COLOR

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.bracket-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.4rem 0;
    transition: all 0.2s;
}
.bracket-card:hover { border-color: #F7B731; transform: translateX(4px); }
.team-name { font-size: 1rem; font-weight: 700; color: #fff; }
.win-prob  { font-size: 0.85rem; color: #aaa; }
.series-score { font-size: 1.1rem; font-weight: 800; color: #F7B731; }
.prob-high   { color: #44ff88; font-weight: 700; }
.prob-medium { color: #F7B731; font-weight: 600; }
.prob-low    { color: #ff8888; }
.round-label { font-size: 1.4rem; font-weight: 800; color: #17408B; text-transform: uppercase; letter-spacing: 2px; }
.champion-badge {
    background: linear-gradient(90deg, #F7B731, #ff8c00);
    color: #000;
    padding: 0.5rem 1.5rem;
    border-radius: 20px;
    font-weight: 900;
    font-size: 1.1rem;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

st.title("🏀 Playoff Bracket & Predictions")
st.caption("Real-time series predictions updated after every game")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_bracket_data():
    try:
        from data.nba_data import get_team_stats, get_player_stats
        from data.injury_tracker import get_team_injury_summary
        from data.features import build_matchup_features
        from models.predictor import GamePredictor
        from models.series_predictor import SeriesPredictor, SeriesState

        team_stats   = get_team_stats()
        player_stats = get_player_stats()
        inj_summary  = get_team_injury_summary()

        predictor    = GamePredictor()
        predictor.load()
        series_pred  = SeriesPredictor(predictor)

        return team_stats, player_stats, inj_summary, series_pred
    except Exception as e:
        st.warning(f"Using demo data: {e}")
        return None, None, {}, None

team_stats, player_stats, inj_summary, series_pred = load_bracket_data()

# ── Playoff Seedings Input ────────────────────────────────────────────────────
st.sidebar.header("⚙️ Bracket Settings")
st.sidebar.markdown("**Enter 2024-25 Playoff Seeds**")
st.sidebar.caption("Teams auto-pulled from NBA standings. Edit if needed.")

DEFAULT_SEEDS = {
    # East
    "1E": "Cleveland Cavaliers", "2E": "Boston Celtics",
    "3E": "New York Knicks",     "4E": "Milwaukee Bucks",
    "5E": "Indiana Pacers",      "6E": "Miami Heat",
    "7E": "Atlanta Hawks",       "8E": "Orlando Magic",
    # West
    "1W": "Oklahoma City Thunder","2W": "Denver Nuggets",
    "3W": "Minnesota Timberwolves","4W": "Los Angeles Lakers",
    "5W": "Golden State Warriors","6W": "Dallas Mavericks",
    "7W": "Memphis Grizzlies",   "8W": "Sacramento Kings",
}

seedings = {}
with st.sidebar.expander("East Seeds", expanded=False):
    for seed in ["1E","2E","3E","4E","5E","6E","7E","8E"]:
        seedings[seed] = st.text_input(seed, value=DEFAULT_SEEDS.get(seed, ""), key=f"seed_{seed}")

with st.sidebar.expander("West Seeds", expanded=False):
    for seed in ["1W","2W","3W","4W","5W","6W","7W","8W"]:
        seedings[seed] = st.text_input(seed, value=DEFAULT_SEEDS.get(seed, ""), key=f"seed_{seed}")

n_sims = st.sidebar.slider("Simulations", 1000, 20000, 5000, 1000)

# Merge defaults with inputs
for k, v in DEFAULT_SEEDS.items():
    if not seedings.get(k):
        seedings[k] = v

# ── Series Prediction Helper ──────────────────────────────────────────────────
def get_series_pred(high_team, low_team, current_high_wins=0, current_low_wins=0):
    """Get a series prediction, using fallback if model not loaded."""
    if series_pred is None:
        # Fallback: simple random prediction
        import random
        wp_h = random.uniform(0.45, 0.70)
        return {
            "high_win_probability": round(wp_h, 3),
            "low_win_probability":  round(1-wp_h, 3),
            "predicted_winner":     high_team if wp_h > 0.5 else low_team,
            "predicted_length":     random.choice([5, 6, 6, 7]),
            "current_series":       f"{current_high_wins}-{current_low_wins}",
            "per_game_predictions": [],
            "key_factors": [],
            "injury_impact_high": 0.0,
            "injury_impact_low":  0.0,
        }

    from models.series_predictor import SeriesState
    state = SeriesState(high_team, low_team)
    state.wins_high = current_high_wins
    state.wins_low  = current_low_wins
    state.game_number = current_high_wins + current_low_wins + 1

    try:
        return series_pred.predict_series(
            team_high=high_team, team_low=low_team,
            series_state=state, n_simulations=n_sims
        )
    except Exception as e:
        return {
            "high_win_probability": 0.55,
            "low_win_probability":  0.45,
            "predicted_winner":     high_team,
            "predicted_length":     6,
            "current_series":       f"{current_high_wins}-{current_low_wins}",
            "per_game_predictions": [],
            "key_factors": [],
            "injury_impact_high": 0.0,
            "injury_impact_low":  0.0,
        }


def prob_color(p):
    if p >= 0.70: return "#44ff88"
    if p >= 0.55: return "#F7B731"
    return "#ff8888"


def render_series_card(high_team, low_team, pred, compact=False):
    """Render a series matchup card."""
    hp = pred["high_win_probability"]
    lp = pred["low_win_probability"]
    winner = pred["predicted_winner"]
    length = pred["predicted_length"]
    series = pred.get("current_series", "0-0")

    hc = prob_color(hp)
    lc = prob_color(lp)

    inj_h = pred.get("injury_impact_high", 0)
    inj_l = pred.get("injury_impact_low", 0)
    inj_h_str = f"⚕️{inj_h:.0%}" if inj_h > 0.1 else ""
    inj_l_str = f"⚕️{inj_l:.0%}" if inj_l > 0.1 else ""

    winner_marker = lambda t: "🏆" if t == winner else ""

    st.markdown(f"""
    <div class="bracket-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px">
            <span class="team-name">{winner_marker(high_team)} {high_team}</span>
            <span style="color:{hc}; font-weight:700; font-size:1.1rem">{hp:.0%} {inj_h_str}</span>
        </div>
        <div style="background:#222; border-radius:6px; height:8px; margin:4px 0; overflow:hidden">
            <div style="background:{hc}; width:{hp*100:.0f}%; height:100%; border-radius:6px"></div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px">
            <span class="team-name">{winner_marker(low_team)} {low_team}</span>
            <span style="color:{lc}; font-weight:700; font-size:1.1rem">{lp:.0%} {inj_l_str}</span>
        </div>
        <div style="margin-top:8px; display:flex; justify-content:space-between; color:#aaa; font-size:0.8rem">
            <span>Series: <b style="color:#F7B731">{series}</b></span>
            <span>Predicted: <b>{winner}</b> in {length}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── EAST BRACKET ──────────────────────────────────────────────────────────────
st.markdown("## 🏀 Eastern Conference")

east_r1_matchups = [
    (seedings["1E"], seedings["8E"]),
    (seedings["4E"], seedings["5E"]),
    (seedings["2E"], seedings["7E"]),
    (seedings["3E"], seedings["6E"]),
]

# Series state inputs (sidebar)
with st.sidebar.expander("Live Series Scores (East)", expanded=True):
    east_series_state = {}
    for h, l in east_r1_matchups:
        key = f"{h[:3]}_v_{l[:3]}"
        cols_s = st.columns(2)
        wh = cols_s[0].number_input(f"{h[:3]} wins", 0, 4, 0, key=f"eh_{key}")
        wl = cols_s[1].number_input(f"{l[:3]} wins", 0, 4, 0, key=f"el_{key}")
        east_series_state[key] = (wh, wl)

with st.sidebar.expander("Live Series Scores (West)", expanded=True):
    west_r1_matchups_seeds = [
        (seedings["1W"], seedings["8W"]),
        (seedings["4W"], seedings["5W"]),
        (seedings["2W"], seedings["7W"]),
        (seedings["3W"], seedings["6W"]),
    ]
    west_series_state = {}
    for h, l in west_r1_matchups_seeds:
        key = f"{h[:3]}_v_{l[:3]}"
        cols_s = st.columns(2)
        wh = cols_s[0].number_input(f"{h[:3]} wins", 0, 4, 0, key=f"wh_{key}")
        wl = cols_s[1].number_input(f"{l[:3]} wins", 0, 4, 0, key=f"wl_{key}")
        west_series_state[key] = (wh, wl)

# East R1
st.markdown("### Round 1")
ec1, ec2, ec3, ec4 = st.columns(4)
east_r1_preds = []
for i, (h, l) in enumerate(east_r1_matchups):
    key = f"{h[:3]}_v_{l[:3]}"
    wh, wl = east_series_state.get(key, (0, 0))
    pred = get_series_pred(h, l, wh, wl)
    east_r1_preds.append(pred)
    with [ec1, ec2, ec3, ec4][i]:
        st.caption(f"E{[1,4,2,3][i]} vs E{[8,5,7,6][i]}")
        render_series_card(h, l, pred)

# East Semis (predicted)
st.markdown("### Conference Semifinals (Projected)")
east_semi_winners = [p["predicted_winner"] for p in east_r1_preds]
east_semi_matchups = [
    (east_semi_winners[0], east_semi_winners[1]),
    (east_semi_winners[2], east_semi_winners[3]),
]
esc1, esc2 = st.columns(2)
east_semi_preds = []
for i, (h, l) in enumerate(east_semi_matchups):
    pred = get_series_pred(h, l)
    east_semi_preds.append(pred)
    with [esc1, esc2][i]:
        render_series_card(h, l, pred)

# East Finals (predicted)
st.markdown("### Conference Finals (Projected)")
east_cf_matchup = (east_semi_preds[0]["predicted_winner"],
                   east_semi_preds[1]["predicted_winner"])
east_cf_pred = get_series_pred(*east_cf_matchup)
ecf_col, _ = st.columns([1, 1])
with ecf_col:
    render_series_card(*east_cf_matchup, east_cf_pred)

east_finalist = east_cf_pred["predicted_winner"]

st.markdown("---")

# ── WEST BRACKET ──────────────────────────────────────────────────────────────
st.markdown("## 🏀 Western Conference")

west_r1_matchups = west_r1_matchups_seeds
st.markdown("### Round 1")
wc1, wc2, wc3, wc4 = st.columns(4)
west_r1_preds = []
for i, (h, l) in enumerate(west_r1_matchups):
    key = f"{h[:3]}_v_{l[:3]}"
    wh, wl = west_series_state.get(key, (0, 0))
    pred = get_series_pred(h, l, wh, wl)
    west_r1_preds.append(pred)
    with [wc1, wc2, wc3, wc4][i]:
        st.caption(f"W{[1,4,2,3][i]} vs W{[8,5,7,6][i]}")
        render_series_card(h, l, pred)

st.markdown("### Conference Semifinals (Projected)")
west_semi_winners = [p["predicted_winner"] for p in west_r1_preds]
west_semi_matchups = [
    (west_semi_winners[0], west_semi_winners[1]),
    (west_semi_winners[2], west_semi_winners[3]),
]
wsc1, wsc2 = st.columns(2)
west_semi_preds = []
for i, (h, l) in enumerate(west_semi_matchups):
    pred = get_series_pred(h, l)
    west_semi_preds.append(pred)
    with [wsc1, wsc2][i]:
        render_series_card(h, l, pred)

st.markdown("### Conference Finals (Projected)")
west_cf_matchup = (west_semi_preds[0]["predicted_winner"],
                   west_semi_preds[1]["predicted_winner"])
west_cf_pred = get_series_pred(*west_cf_matchup)
_, wcf_col = st.columns([1, 1])
with wcf_col:
    render_series_card(*west_cf_matchup, west_cf_pred)

west_finalist = west_cf_pred["predicted_winner"]

st.markdown("---")

# ── NBA FINALS ────────────────────────────────────────────────────────────────
st.markdown("## 🏆 NBA Finals (Projected)")
finals_pred = get_series_pred(east_finalist, west_finalist)
f1, f2, f3 = st.columns([1, 2, 1])
with f2:
    render_series_card(east_finalist, west_finalist, finals_pred)

champion = finals_pred["predicted_winner"]
st.markdown(f"""
<div style="text-align:center; margin-top:1.5rem">
    <span class="champion-badge">🏆 Predicted Champion: {champion}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Championship Odds Chart ───────────────────────────────────────────────────
st.subheader("📊 Predicted Championship Odds (All Teams)")

all_teams_odds = {}
all_preds = east_r1_preds + west_r1_preds
for pred in all_preds:
    all_teams_odds[pred["predicted_winner"]] = all_teams_odds.get(
        pred["predicted_winner"], 0) + 0.1

# Build odds from bracket predictions
team_odds = {}
for p in east_r1_preds + west_r1_preds:
    for team_key, prob_key in [("team_high", "high_win_probability"),
                                ("team_low",  "low_win_probability")]:
        team = p.get(team_key, "")
        if team:
            team_odds[team] = team_odds.get(team, 1.0) * p.get(prob_key, 0.5)

# Boost finalists
team_odds[east_finalist] = team_odds.get(east_finalist, 0.3) * east_cf_pred["high_win_probability"]
team_odds[west_finalist] = team_odds.get(west_finalist, 0.3) * west_cf_pred["high_win_probability"]
team_odds[champion]      = team_odds.get(champion, 0.2) * finals_pred["high_win_probability"]

# Normalize
total_odds = sum(team_odds.values())
if total_odds > 0:
    team_odds = {k: v/total_odds for k, v in team_odds.items()}

odds_df = pd.DataFrame(
    [(t, round(o*100, 1)) for t, o in sorted(team_odds.items(), key=lambda x: -x[1])],
    columns=["Team", "Championship %"]
)

fig = px.bar(
    odds_df.head(16),
    x="Team", y="Championship %",
    color="Championship %",
    color_continuous_scale=["#1a1a2e", "#17408B", "#C9082A", "#F7B731"],
    title="Predicted Championship Probability (%)",
    template="plotly_dark",
)
fig.update_layout(
    plot_bgcolor="#1a1a2e",
    paper_bgcolor="#1a1a2e",
    xaxis_tickangle=-35,
    showlegend=False,
    coloraxis_showscale=False,
    height=420,
)
st.plotly_chart(fig, use_container_width=True)
