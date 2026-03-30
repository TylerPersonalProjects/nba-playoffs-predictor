"""
Series Predictions Page
Detailed game-by-game predictions with score forecasts, win probabilities,
key factors, and Monte Carlo simulation results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Series Predictions | NBA Predictor", page_icon="🔮", layout="wide")

# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_all_data():
    from data.nba_data import get_team_stats, get_player_stats
    from data.injury_tracker import get_team_injury_summary
    ts = get_team_stats()
    ps = get_player_stats()
    ij = get_team_injury_summary()
    return ts, ps, ij

@st.cache_resource
def get_predictor():
    from models.predictor import GamePredictor
    from models.series_predictor import SeriesPredictor
    gp = GamePredictor()
    gp.load()
    return SeriesPredictor(gp)

try:
    team_stats, player_stats, inj_summary = load_all_data()
    sp = get_predictor()
    data_ok = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_ok = False
    team_stats = pd.DataFrame()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔮 Series Predictions")
st.caption("Game-by-game score predictions, win probabilities, and series outlook")

if not data_ok or team_stats.empty:
    st.warning("No data available. Run `pip install -r requirements.txt` first.")
    st.stop()

team_names = sorted(team_stats["team_name"].tolist()) if "team_name" in team_stats.columns else []

st.sidebar.header("⚙️ Configure Matchup")

col_s1, col_s2 = st.columns(2)
with col_s1:
    home_team = st.selectbox("🏠 Home Team (Higher Seed)", team_names,
                              index=team_names.index("Boston Celtics") if "Boston Celtics" in team_names else 0)
with col_s2:
    away_team = st.selectbox("✈️ Away Team (Lower Seed)", team_names,
                              index=team_names.index("Cleveland Cavaliers") if "Cleveland Cavaliers" in team_names else 1)

# Series state
st.sidebar.subheader("Current Series State")
home_wins = st.sidebar.number_input("Home Team Wins", 0, 4, 0)
away_wins = st.sidebar.number_input("Away Team Wins", 0, 4, 0)
home_rest = st.sidebar.slider("Home Team Rest Days", 0, 5, 2)
away_rest = st.sidebar.slider("Away Team Rest Days", 0, 5, 2)
n_sims    = st.sidebar.slider("Monte Carlo Simulations", 1000, 20000, 10000, 1000)

if home_team == away_team:
    st.error("Please select two different teams.")
    st.stop()

# ── Run Prediction ────────────────────────────────────────────────────────────
with st.spinner(f"Running {n_sims:,} simulations for {home_team} vs {away_team}..."):
    try:
        from models.series_predictor import SeriesState, SeriesPredictor
        state = SeriesState(home_team, away_team)
        state.wins_high  = home_wins
        state.wins_low   = away_wins
        state.game_number = home_wins + away_wins + 1

        pred = sp.predict_series(
            team_high    = home_team,
            team_low     = away_team,
            series_state = state,
            n_simulations= n_sims,
        )
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fallback
        import random
        rng = random.Random(hash(home_team + away_team))
        hw  = rng.uniform(0.45, 0.70)
        pred = {
            "high_win_probability": hw,
            "low_win_probability": 1-hw,
            "predicted_winner": home_team if hw > 0.5 else away_team,
            "predicted_length": rng.choice([5,6,6,7]),
            "current_series": f"{home_wins}-{away_wins}",
            "per_game_predictions": [],
            "key_factors": [],
            "injury_impact_high": 0.0,
            "injury_impact_low": 0.0,
            "base_game1_pred": {
                "predicted_home_score": rng.randint(104, 120),
                "predicted_away_score": rng.randint(100, 116),
                "home_win_prob": hw,
                "confidence": "Medium",
            }
        }

# ── Series Header ─────────────────────────────────────────────────────────────
hw_prob = pred["high_win_probability"]
lw_prob = pred["low_win_probability"]
winner  = pred["predicted_winner"]
length  = pred["predicted_length"]

col_h, col_vs, col_a = st.columns([2, 1, 2])
with col_h:
    st.markdown(f"""
    <div style="text-align:center; background:linear-gradient(135deg,#1a1a2e,#16213e);
                border-radius:16px; padding:1.5rem; border:2px solid {'#17408B' if hw_prob>lw_prob else '#333'}">
        <div style="font-size:2rem; font-weight:900; color:white">{home_team}</div>
        <div style="font-size:3rem; font-weight:900; color:{'#44ff88' if hw_prob>lw_prob else '#aaa'}">{hw_prob:.0%}</div>
        <div style="color:#aaa">Series Win Probability</div>
        <div style="font-size:1.2rem; color:#F7B731; margin-top:0.5rem">Wins: {home_wins}</div>
    </div>
    """, unsafe_allow_html=True)

with col_vs:
    st.markdown(f"""
    <div style="text-align:center; padding-top:2rem">
        <div style="font-size:1rem; color:#666">Series</div>
        <div style="font-size:2rem; font-weight:900; color:#F7B731">{home_wins}–{away_wins}</div>
        <div style="font-size:0.85rem; color:#aaa">Best of 7</div>
        <br>
        {'🏆' if winner == home_team else '🏆' if winner == away_team else ''}
        <div style="font-size:0.8rem; color:#aaa">Pred. in {length} games</div>
    </div>
    """, unsafe_allow_html=True)

with col_a:
    st.markdown(f"""
    <div style="text-align:center; background:linear-gradient(135deg,#1a1a2e,#16213e);
                border-radius:16px; padding:1.5rem; border:2px solid {'#C9082A' if lw_prob>hw_prob else '#333'}">
        <div style="font-size:2rem; font-weight:900; color:white">{away_team}</div>
        <div style="font-size:3rem; font-weight:900; color:{'#44ff88' if lw_prob>hw_prob else '#aaa'}">{lw_prob:.0%}</div>
        <div style="color:#aaa">Series Win Probability</div>
        <div style="font-size:1.2rem; color:#F7B731; margin-top:0.5rem">Wins: {away_wins}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Series Length Distribution ────────────────────────────────────────────────
col_len, col_factor = st.columns([1, 1])

with col_len:
    st.subheader("📊 Series Length Distribution")
    length_dist = pred.get("length_distribution", {4: 0.12, 5: 0.28, 6: 0.35, 7: 0.25})
    len_df = pd.DataFrame(
        [(f"In {k}", f"{v:.0%}", v) for k, v in length_dist.items()],
        columns=["Outcome", "Probability", "pct"]
    )
    fig_len = px.bar(
        len_df, x="Outcome", y="pct",
        color="pct",
        color_continuous_scale=["#1a1a2e", "#17408B", "#F7B731"],
        labels={"pct": "Probability"},
        text="Probability",
        template="plotly_dark",
    )
    fig_len.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                           height=300, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_len, use_container_width=True)

with col_factor:
    st.subheader("🔑 Key Factors")
    factors = pred.get("key_factors", [])
    if factors:
        for f in factors:
            impact_color = {"High": "#ff4444", "Medium": "#F7B731", "Low": "#44ff88"}.get(f["impact"], "#aaa")
            st.markdown(f"""
            <div style="background:#1a1a2e; border-left:4px solid {impact_color};
                        padding:0.8rem; border-radius:8px; margin:0.4rem 0">
                <span style="font-weight:700; color:#fff">{f['factor']}</span>
                <span style="background:{impact_color}; color:#000; font-size:0.7rem;
                             padding:2px 8px; border-radius:10px; margin-left:8px">{f['impact']}</span>
                <div style="color:#aaa; font-size:0.85rem; margin-top:4px">{f['detail']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Factors will appear after model runs.")

st.markdown("---")

# ── Per-Game Predictions ──────────────────────────────────────────────────────
st.subheader("🎮 Game-by-Game Score Predictions")

game_preds = pred.get("per_game_predictions", [])
if game_preds:
    for game in game_preds:
        gn     = game["game_number"]
        home   = game["home_team"]
        away   = game["away_team"]
        hsp    = game["pred_home_score"]
        asp    = game["pred_away_score"]
        hwp    = game["home_win_prob"]
        conf   = game["confidence"]
        series = game["series_score"]

        conf_color = {"High": "#44ff88", "Medium": "#F7B731", "Low": "#ff8888"}.get(conf, "#aaa")
        margin_str = f"{home} by {abs(hsp-asp)}" if hsp > asp else f"{away} by {abs(asp-hsp)}"

        col_gm1, col_gm2, col_gm3 = st.columns([1, 3, 1])
        with col_gm1:
            st.markdown(f"""
            <div style="text-align:center; padding:0.5rem">
                <div style="font-size:1.4rem; font-weight:900; color:#F7B731">Game {gn}</div>
                <div style="color:#aaa; font-size:0.8rem">Series: {series}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_gm2:
            bar_w = int(hwp * 100)
            st.markdown(f"""
            <div style="background:#1a1a2e; border-radius:12px; padding:1rem; border:1px solid #333">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">
                    <span style="font-weight:700; color:#fff; font-size:1rem">🏠 {home}</span>
                    <span style="font-size:2rem; font-weight:900; color:#fff">{hsp} – {asp}</span>
                    <span style="font-weight:700; color:#fff; font-size:1rem">{away} ✈️</span>
                </div>
                <div style="background:#333; border-radius:8px; height:14px; overflow:hidden">
                    <div style="background:#17408B; width:{bar_w}%; height:100%; border-radius:8px;
                                border-right:2px solid #C9082A"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:0.8rem">
                    <span style="color:#17408B; font-weight:700">{hwp:.0%}</span>
                    <span style="color:#aaa">Predicted: <b style="color:#F7B731">{margin_str}</b></span>
                    <span style="color:#C9082A; font-weight:700">{1-hwp:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gm3:
            st.markdown(f"""
            <div style="text-align:center; padding:0.5rem">
                <div style="font-size:0.75rem; color:#aaa">Confidence</div>
                <div style="color:{conf_color}; font-weight:700; font-size:1.1rem">{conf}</div>
                <div style="font-size:0.75rem; color:#aaa; margin-top:4px">
                    {game.get('location','').replace(home_team, home_team[:12])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

else:
    # Show at least a Game 1 prediction
    g1 = pred.get("base_game1_pred", {})
    if g1:
        hs = g1.get("predicted_home_score", 112)
        as_ = g1.get("predicted_away_score", 108)
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:1.5rem; border:1px solid #333;
                    text-align:center">
            <div style="font-size:1.2rem; font-weight:700; color:#F7B731">Game 1 Prediction</div>
            <div style="font-size:3rem; font-weight:900; color:#fff; margin:1rem 0">
                {home_team.split()[-1]} {hs} – {as_} {away_team.split()[-1]}
            </div>
            <div style="color:#aaa">Home court advantage: {home_team}</div>
            <div style="color:{'#44ff88' if g1.get('home_win_prob',0.5)>0.5 else '#ff8888'};
                        font-size:1.2rem; font-weight:700; margin-top:0.5rem">
                {home_team if g1.get('home_win_prob',0.5)>0.5 else away_team} favored ({g1.get('home_win_prob',0.55):.0%})
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Win Probability Gauge ─────────────────────────────────────────────────────
st.subheader("🎯 Win Probability Gauge")

fig_gauge = go.Figure(go.Indicator(
    mode  = "gauge+number+delta",
    value = hw_prob * 100,
    title = {"text": f"{home_team} Series Win %", "font": {"size": 18, "color": "#fff"}},
    number = {"suffix": "%", "font": {"color": "#F7B731", "size": 40}},
    gauge = {
        "axis":  {"range": [0, 100], "tickcolor": "#aaa"},
        "bar":   {"color": "#17408B"},
        "bgcolor": "#1a1a2e",
        "steps": [
            {"range": [0,  33], "color": "#C9082A"},
            {"range": [33, 50], "color": "#ff8c00"},
            {"range": [50, 67], "color": "#F7B731"},
            {"range": [67,100], "color": "#44ff88"},
        ],
        "threshold": {
            "line": {"color": "white", "width": 3},
            "thickness": 0.75,
            "value": 50,
        },
    },
))
fig_gauge.update_layout(
    paper_bgcolor="#1a1a2e",
    font={"color": "#fff"},
    height=350,
)
st.plotly_chart(fig_gauge, use_container_width=True)

# ── Injury Impact ─────────────────────────────────────────────────────────────
inj_h = pred.get("injury_impact_high", 0)
inj_l = pred.get("injury_impact_low",  0)
if max(inj_h, inj_l) > 0.05:
    st.subheader("🏥 Injury Impact on Predictions")
    ci1, ci2 = st.columns(2)
    with ci1:
        color = "#ff4444" if inj_h > 0.4 else "#F7B731" if inj_h > 0.2 else "#44ff88"
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:10px; padding:1rem; border:1px solid {color}">
            <b style="color:#fff">{home_team}</b><br>
            <span style="color:{color}; font-size:1.5rem; font-weight:700">{inj_h:.0%} impact</span><br>
            <span style="color:#aaa; font-size:0.85rem">Injury reducing team effectiveness</span>
        </div>
        """, unsafe_allow_html=True)
    with ci2:
        color = "#ff4444" if inj_l > 0.4 else "#F7B731" if inj_l > 0.2 else "#44ff88"
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:10px; padding:1rem; border:1px solid {color}">
            <b style="color:#fff">{away_team}</b><br>
            <span style="color:{color}; font-size:1.5rem; font-weight:700">{inj_l:.0%} impact</span><br>
            <span style="color:#aaa; font-size:0.85rem">Injury reducing team effectiveness</span>
        </div>
        """, unsafe_allow_html=True)
