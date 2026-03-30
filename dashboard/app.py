"""
NBA Playoffs Predictor — Main Streamlit Dashboard
Entry point: streamlit run dashboard/app.py
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DASHBOARD_TITLE, REFRESH_INTERVAL_MS, PRIMARY_COLOR, SECONDARY_COLOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title    = "NBA Playoffs Predictor",
    page_icon     = "🏀",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Main theme */
    :root {{
        --primary:   {PRIMARY_COLOR};
        --secondary: {SECONDARY_COLOR};
        --accent:    #F7B731;
        --bg-card:   #1a1a2e;
        --bg-dark:   #16213e;
    }}

    .main {{ background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }}

    /* Header */
    .hero-title {{
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #17408B, #C9082A, #F7B731);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }}
    .hero-subtitle {{
        text-align: center;
        color: #aaa;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }}

    /* Metric cards */
    .metric-card {{
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
    }}
    .metric-card:hover {{ transform: translateY(-3px); }}
    .metric-value {{
        font-size: 2.4rem;
        font-weight: 800;
        color: #F7B731;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* Team pill */
    .team-pill {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background: {PRIMARY_COLOR}33;
        border: 1px solid {PRIMARY_COLOR};
        color: #fff;
        font-size: 0.85rem;
        margin: 2px;
    }}

    /* Probability bar */
    .prob-bar-container {{
        background: #222;
        border-radius: 8px;
        height: 22px;
        overflow: hidden;
        margin: 8px 0;
    }}

    /* Injury badge */
    .injury-critical {{ color: #ff4444; font-weight: 700; }}
    .injury-high     {{ color: #ff8c00; font-weight: 600; }}
    .injury-moderate {{ color: #ffd700; }}
    .injury-none     {{ color: #44ff88; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #333;
    }}

    /* Live badge */
    .live-badge {{
        display: inline-block;
        background: #ff4444;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 10px;
        animation: pulse 1.5s infinite;
        margin-left: 8px;
    }}
    @keyframes pulse {{
        0%   {{ opacity: 1; }}
        50%  {{ opacity: 0.5; }}
        100% {{ opacity: 1; }}
    }}

    /* Score display */
    .score-display {{
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        color: white;
    }}
    .score-vs {{
        font-size: 1rem;
        color: #666;
        text-align: center;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #555;
        font-size: 0.75rem;
        padding: 1rem;
        border-top: 1px solid #222;
        margin-top: 2rem;
    }}

    /* Hide streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer    {{ visibility: hidden; }}
    header    {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Session state initialization ──────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "injury_df_snapshot" not in st.session_state:
    st.session_state.injury_df_snapshot = None
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">🏀 NBA Playoffs Predictor</div>
<div class="hero-subtitle">Real-Time Machine Learning · Powered by nba_api · Updated Every 5 Minutes</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Top Metrics Row ───────────────────────────────────────────────────────────
try:
    from data.nba_data import get_team_stats, get_player_stats, get_live_scores
    from data.injury_tracker import get_injury_report

    with st.spinner("Loading live data..."):
        team_stats   = get_team_stats()
        player_stats = get_player_stats()
        injury_df    = get_injury_report()
        live_games   = get_live_scores()

    n_teams   = len(team_stats) if not team_stats.empty else 30
    n_players = len(player_stats) if not player_stats.empty else 450
    n_injured = len(injury_df) if not injury_df.empty else 0
    n_live    = len(live_games)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_teams}</div>
            <div class="metric-label">Teams Tracked</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_players}</div>
            <div class="metric-label">Players Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_injured}</div>
            <div class="metric-label">Injured Players</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_live}</div>
            <div class="metric-label">{'Live Games' if n_live > 0 else 'Games Today'}</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        now = datetime.now().strftime("%I:%M %p")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem">{now}</div>
            <div class="metric-label">Last Updated</div>
        </div>""", unsafe_allow_html=True)

except Exception as e:
    st.warning(f"⚠️ Could not load live data: {e}. Install dependencies and ensure network access.")
    st.info("Run `pip install -r requirements.txt` then `streamlit run dashboard/app.py`")

st.markdown("---")

# ── Live Games ────────────────────────────────────────────────────────────────
try:
    if live_games:
        st.subheader("🔴 Live & Today's Games" + ' <span class="live-badge">LIVE</span>' if any(
            "Q" in g.get("status", "") or "Half" in g.get("status", "") for g in live_games
        ) else "📅 Today's Games")

        cols = st.columns(min(len(live_games), 4))
        for i, game in enumerate(live_games[:4]):
            with cols[i % 4]:
                status_color = "#ff4444" if "Q" in game.get("status", "") else "#aaa"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color:{status_color}; font-size:0.8rem; font-weight:700; text-transform:uppercase">
                        {game.get('status','Final')}
                    </div>
                    <div class="score-display">
                        {game.get('away_team','')} {game.get('away_score',0)}
                    </div>
                    <div class="score-vs">@ {game.get('home_team','')}</div>
                    <div class="score-display">
                        {game.get('home_score',0)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")
except Exception:
    pass

# ── Navigation Guide ──────────────────────────────────────────────────────────
st.subheader("📍 Navigate the Dashboard")

nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    st.markdown("""
    **🏆 Pages (Sidebar)**

    - 🏀 **Bracket** — Full playoff bracket with series predictions and win probabilities
    - 📊 **Team Analysis** — Deep dive: offense, defense, pace, and trends
    - 🔮 **Series Predictions** — Game-by-game predictions with score forecasts
    """)
with nav_col2:
    st.markdown("""
    **📈 More Pages**

    - 👤 **Player Stats** — Top performers, usage rates, efficiency ratings
    - 🏥 **Injury Tracker** — Live injury updates with team impact scores
    """)
with nav_col3:
    st.markdown("""
    **⚙️ How It Works**

    The model uses **XGBoost + Random Forest + Logistic Regression** ensemble trained on 5 seasons of NBA data.

    Features: Offensive/Defensive Rating, Pace, eFG%, Injury Impact, Rest Days, H2H Records, Recent Form.

    Predictions update every **5 minutes** automatically.
    """)

st.markdown("---")

# ── Quick Standings Snapshot ──────────────────────────────────────────────────
try:
    if not team_stats.empty:
        st.subheader("📊 Top Teams This Season")
        display_cols = ["team_name", "wins", "losses", "win_pct",
                        "pts_per_game", "net_rating", "off_rating", "def_rating"]
        available = [c for c in display_cols if c in team_stats.columns]

        top_teams = team_stats.sort_values("net_rating", ascending=False).head(10)[available]
        top_teams.columns = [c.replace("_", " ").title() for c in top_teams.columns]

        st.dataframe(
            top_teams.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
except Exception:
    pass

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown("---")
last = datetime.fromtimestamp(st.session_state.last_refresh).strftime("%I:%M:%S %p")
st.caption(f"🔄 Auto-refreshes every 5 minutes | Last data pull: {last}")

# Streamlit rerun trigger
st.markdown(
    f"""<meta http-equiv="refresh" content="{REFRESH_INTERVAL_MS // 1000}">""",
    unsafe_allow_html=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    NBA Playoffs Predictor &nbsp;|&nbsp; Data: nba_api + ESPN Injury Reports &nbsp;|&nbsp;
    Model: XGBoost + Random Forest Ensemble &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
