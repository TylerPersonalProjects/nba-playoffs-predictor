"""
Player Stats Page
Top performers, usage rates, efficiency, and star player impact analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Player Stats | NBA Predictor", page_icon="👤", layout="wide")

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_players():
    from data.nba_data import get_player_stats
    return get_player_stats()

try:
    player_stats = load_players()
    data_ok = not player_stats.empty
except Exception as e:
    st.error(f"Data error: {e}")
    data_ok = False
    player_stats = pd.DataFrame()

st.title("👤 Player Statistics & Impact")
st.caption("Scoring leaders, efficiency ratings, and star player analysis")

if not data_ok:
    st.warning("No player data available.")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    min_games = st.slider("Min Games Played", 10, 60, 30)
with col_f2:
    stat_filter = st.selectbox("Sort By", ["pts_per_game", "reb_per_game", "ast_per_game",
                                            "ts_pct", "net_rating", "usg_pct", "per_approx",
                                            "efg_pct", "blk_per_game", "stl_per_game"])
with col_f3:
    if "team_abbr" in player_stats.columns:
        team_filter = st.selectbox("Team Filter", ["All Teams"] + sorted(player_stats["team_abbr"].dropna().unique().tolist()))
    else:
        team_filter = "All Teams"

# Apply filters
df = player_stats.copy()
if "games_played" in df.columns:
    df = df[df["games_played"] >= min_games]
if team_filter != "All Teams" and "team_abbr" in df.columns:
    df = df[df["team_abbr"] == team_filter]
if stat_filter in df.columns:
    df = df.sort_values(stat_filter, ascending=False)

st.markdown(f"**{len(df)} players** match your filters")

# ── Top Stats Cards ───────────────────────────────────────────────────────────
st.subheader("🏅 League Leaders")
leader_cols = st.columns(4)

stats_to_show = [
    ("pts_per_game",  "Scoring Leader",   "PPG"),
    ("reb_per_game",  "Rebounding Leader","RPG"),
    ("ast_per_game",  "Assist Leader",    "APG"),
    ("ts_pct",        "TS% Leader",       "TS%"),
]

for i, (col_name, title, unit) in enumerate(stats_to_show):
    if col_name in df.columns and len(df) > 0:
        top_player = df.loc[df[col_name].idxmax()]
        val = float(top_player[col_name])
        name = str(top_player.get("player_name", "Unknown"))
        team = str(top_player.get("team_abbr", ""))
        display_val = f"{val:.1%}" if "pct" in col_name else f"{val:.1f}"
        with leader_cols[i]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                        border:1px solid #333; border-radius:12px; padding:1rem; text-align:center">
                <div style="color:#aaa; font-size:0.75rem; text-transform:uppercase">{title}</div>
                <div style="font-size:1.1rem; font-weight:700; color:#fff; margin:4px 0">{name}</div>
                <div style="color:#666; font-size:0.8rem">{team}</div>
                <div style="font-size:2rem; font-weight:900; color:#F7B731">{display_val}</div>
                <div style="color:#aaa; font-size:0.7rem">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Scatter: Scoring vs Efficiency ────────────────────────────────────────────
st.subheader("📈 Scoring vs. Efficiency Scatter")

if "pts_per_game" in df.columns and "ts_pct" in df.columns:
    scatter_df = df.dropna(subset=["pts_per_game", "ts_pct"]).head(100)
    fig_scatter = px.scatter(
        scatter_df,
        x="pts_per_game",
        y="ts_pct",
        size="min_per_game" if "min_per_game" in scatter_df else None,
        color="usg_pct" if "usg_pct" in scatter_df else None,
        hover_name="player_name" if "player_name" in scatter_df else None,
        hover_data=["team_abbr", "pts_per_game", "ts_pct",
                    "usg_pct"] if all(c in scatter_df for c in ["team_abbr","usg_pct"]) else None,
        labels={"pts_per_game": "Points Per Game", "ts_pct": "True Shooting %",
                "usg_pct": "Usage Rate", "min_per_game": "Minutes"},
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Scoring Volume vs. Shooting Efficiency (size = minutes played)",
    )
    fig_scatter.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Top 25 Table ──────────────────────────────────────────────────────────────
st.subheader(f"📋 Top 25 Players by {stat_filter.replace('_per_game','').replace('_',' ').upper()}")

display_cols = ["player_name", "team_abbr", "games_played", "min_per_game",
                "pts_per_game", "reb_per_game", "ast_per_game",
                "stl_per_game", "blk_per_game", "fg_pct", "fg3_pct",
                "ts_pct", "efg_pct", "usg_pct", "net_rating", "per_approx"]
avail = [c for c in display_cols if c in df.columns]
top25 = df[avail].head(25).reset_index(drop=True)

# Format column names
top25.columns = [c.replace("_per_game","").replace("_"," ").upper() for c in top25.columns]

# Style
pct_cols = [c for c in top25.columns if "PCT" in c or "TS" in c or "EFG" in c or "USG" in c]
num_cols  = [c for c in top25.columns if c not in ["PLAYER NAME","TEAM ABBR"]]

st.dataframe(
    top25.style
    .background_gradient(subset=[c for c in ["PTS","NET RATING","TS PCT","PER APPROX"]
                                   if c in top25.columns], cmap="Blues")
    .format({c: "{:.1%}" for c in pct_cols if c in top25.columns})
    .format({c: "{:.1f}" for c in num_cols if c not in pct_cols and c in top25.columns}),
    use_container_width=True,
    hide_index=True,
    height=600,
)

st.markdown("---")

# ── Usage vs Net Rating ───────────────────────────────────────────────────────
st.subheader("⚙️ Usage Rate vs. Net Rating (Star Player Map)")
if "usg_pct" in df.columns and "net_rating" in df.columns:
    star_df = df.dropna(subset=["usg_pct","net_rating","pts_per_game"]).head(80)
    fig_star = px.scatter(
        star_df,
        x="usg_pct",
        y="net_rating",
        size="pts_per_game",
        color="pts_per_game",
        hover_name="player_name" if "player_name" in star_df else None,
        hover_data=["team_abbr","pts_per_game","usg_pct","net_rating"]
                   if all(c in star_df for c in ["team_abbr","usg_pct"]) else None,
        labels={"usg_pct": "Usage Rate", "net_rating": "Net Rating",
                "pts_per_game": "PPG"},
        color_continuous_scale=["#1a1a2e","#17408B","#F7B731","#C9082A"],
        template="plotly_dark",
        title="Who carries the load AND impacts winning? (size = PPG)",
    )
    fig_star.add_hline(y=0, line_dash="dash", line_color="#555", annotation_text="Neutral")
    fig_star.add_vline(x=0.25, line_dash="dash", line_color="#555", annotation_text="25% Usage")
    fig_star.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", height=450)
    st.plotly_chart(fig_star, use_container_width=True)

# ── Player Impact on Predictions ──────────────────────────────────────────────
st.subheader("💥 Star Player Impact on Series Outcome")
st.caption("How much does each team's star losing affect their playoff win probability?")

if "pts_per_game" in df.columns and "team_abbr" in df.columns:
    # Group by team — find top scorer
    top_scorers = df.groupby("team_abbr", group_keys=False).apply(
        lambda x: x.nlargest(1, "pts_per_game")
    ).reset_index(drop=True)

    if not top_scorers.empty:
        fig_impact = px.bar(
            top_scorers.sort_values("pts_per_game", ascending=False).head(16),
            x="team_abbr" if "team_abbr" in top_scorers else "player_name",
            y="pts_per_game",
            color="usg_pct" if "usg_pct" in top_scorers else "pts_per_game",
            hover_name="player_name" if "player_name" in top_scorers else None,
            color_continuous_scale="Oranges",
            labels={"pts_per_game": "Star Player PPG", "team_abbr": "Team",
                    "usg_pct": "Usage%"},
            template="plotly_dark",
            title="Top Scorer Per Team (colored by usage rate)",
        )
        fig_impact.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                                   height=380, coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_impact, use_container_width=True)
