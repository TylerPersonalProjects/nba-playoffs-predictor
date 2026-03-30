"""
Team Analysis Page
Deep-dive into any team: offense, defense, pace, player breakdown, trends.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Team Analysis | NBA Predictor", page_icon="📊", layout="wide")

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    from data.nba_data import get_team_stats, get_player_stats, get_recent_form
    from data.injury_tracker import get_injury_report
    ts = get_team_stats()
    ps = get_player_stats()
    ir = get_injury_report()
    return ts, ps, ir

try:
    team_stats, player_stats, injury_df = load_data()
    data_ok = not team_stats.empty
except Exception as e:
    st.error(f"Data error: {e}")
    data_ok = False
    team_stats = pd.DataFrame()
    player_stats = pd.DataFrame()
    injury_df = pd.DataFrame()

# ── Team Selector ─────────────────────────────────────────────────────────────
st.title("📊 Team Analysis")
st.caption("Full offensive, defensive, and situational breakdown for any NBA team")

if not data_ok:
    st.warning("Could not load team data. Make sure nba_api is installed and you have internet access.")
    st.stop()

team_names = sorted(team_stats["team_name"].tolist()) if "team_name" in team_stats.columns else []
if not team_names:
    st.warning("No team data available.")
    st.stop()

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    selected_team = st.selectbox("Select Team", team_names, index=0)
with col_sel2:
    compare_team  = st.selectbox("Compare Against (optional)", ["None"] + team_names, index=0)

# ── Get team row ──────────────────────────────────────────────────────────────
def get_team_row(name):
    match = team_stats[team_stats["team_name"] == name]
    return match.iloc[0] if not match.empty else None

team_row = get_team_row(selected_team)
if team_row is None:
    st.error(f"No data for {selected_team}")
    st.stop()

compare_row = get_team_row(compare_team) if compare_team != "None" else None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"## 🏀 {selected_team}")

wins = int(team_row.get("wins", 0))
losses = int(team_row.get("losses", 0))
net = float(team_row.get("net_rating", 0))
off = float(team_row.get("off_rating", 112))
def_ = float(team_row.get("def_rating", 112))
pace = float(team_row.get("pace", 99))
pts  = float(team_row.get("pts_per_game", 112))

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Record",        f"{wins}-{losses}")
col2.metric("Net Rating",    f"{net:+.1f}", delta=f"#{int(team_stats['net_rating'].rank(ascending=False)[team_stats['team_name']==selected_team].values[0])} in NBA" if 'net_rating' in team_stats else None)
col3.metric("Off Rating",    f"{off:.1f}")
col4.metric("Def Rating",    f"{def_:.1f}")
col5.metric("Pace",          f"{pace:.1f}")
col6.metric("PPG",           f"{pts:.1f}")

st.markdown("---")

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.subheader("🕸️ Team Profile Radar")

radar_metrics = {
    "Scoring":   min(float(team_row.get("pts_per_game", 112)) / 130, 1.0),
    "Defense":   max(1 - (float(team_row.get("def_rating", 112)) - 100) / 25, 0),
    "Rebounding":min(float(team_row.get("reb_per_game", 44)) / 55, 1.0),
    "Assists":   min(float(team_row.get("ast_per_game", 26)) / 35, 1.0),
    "Shooting":  min(float(team_row.get("efg_pct", 0.54)) / 0.65, 1.0),
    "Pace":      min((float(team_row.get("pace", 99)) - 93) / 12, 1.0),
    "Steals":    min(float(team_row.get("stl_per_game", 7)) / 12, 1.0),
    "Blocks":    min(float(team_row.get("blk_per_game", 5)) / 9, 1.0),
}

cats   = list(radar_metrics.keys())
vals   = list(radar_metrics.values())
vals  += [vals[0]]  # close the loop
cats  += [cats[0]]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=vals, theta=cats,
    fill="toself",
    fillcolor=f"rgba(23, 64, 139, 0.4)",
    line=dict(color="#17408B", width=2),
    name=selected_team,
))

if compare_row is not None:
    cmp_vals = {
        "Scoring":   min(float(compare_row.get("pts_per_game", 112)) / 130, 1.0),
        "Defense":   max(1 - (float(compare_row.get("def_rating", 112)) - 100) / 25, 0),
        "Rebounding":min(float(compare_row.get("reb_per_game", 44)) / 55, 1.0),
        "Assists":   min(float(compare_row.get("ast_per_game", 26)) / 35, 1.0),
        "Shooting":  min(float(compare_row.get("efg_pct", 0.54)) / 0.65, 1.0),
        "Pace":      min((float(compare_row.get("pace", 99)) - 93) / 12, 1.0),
        "Steals":    min(float(compare_row.get("stl_per_game", 7)) / 12, 1.0),
        "Blocks":    min(float(compare_row.get("blk_per_game", 5)) / 9, 1.0),
    }
    cv = list(cmp_vals.values()) + [list(cmp_vals.values())[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=cv, theta=cats,
        fill="toself",
        fillcolor="rgba(201, 8, 42, 0.3)",
        line=dict(color="#C9082A", width=2),
        name=compare_team,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
    showlegend=True,
    template="plotly_dark",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    height=420,
    title="Team Profile (normalized 0-1)",
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Offensive vs Defensive Breakdown ─────────────────────────────────────────
st.subheader("⚔️ Offense vs Defense")
tab_off, tab_def, tab_misc = st.tabs(["Offense", "Defense", "Advanced"])

with tab_off:
    off_metrics = {
        "PPG":      float(team_row.get("pts_per_game", 0)),
        "FG%":      float(team_row.get("fg_pct", 0)) * 100,
        "3P%":      float(team_row.get("fg3_pct", 0)) * 100,
        "FT%":      float(team_row.get("ft_pct", 0)) * 100,
        "APG":      float(team_row.get("ast_per_game", 0)),
        "OREB":     float(team_row.get("oreb_per_game", 0)),
        "eFG%":     float(team_row.get("efg_pct", 0)) * 100,
        "TS%":      float(team_row.get("ts_pct", 0)) * 100,
        "Off Rtg":  float(team_row.get("off_rating", 0)),
    }

    league_avg = {
        "PPG": 113.7, "FG%": 46.3, "3P%": 36.1, "FT%": 77.9,
        "APG": 25.9, "OREB": 10.1, "eFG%": 53.5, "TS%": 58.0, "Off Rtg": 114.0,
    }

    rows = [{"Metric": k, "Team": v, "League Avg": league_avg.get(k, v),
             "vs League": v - league_avg.get(k, v)} for k, v in off_metrics.items()]
    off_df = pd.DataFrame(rows)

    fig_off = go.Figure()
    fig_off.add_trace(go.Bar(name=selected_team, x=off_df["Metric"], y=off_df["Team"],
                              marker_color="#17408B"))
    if compare_row:
        cmp_off = {
            "PPG":  float(compare_row.get("pts_per_game",0)),
            "FG%":  float(compare_row.get("fg_pct",0))*100,
            "3P%":  float(compare_row.get("fg3_pct",0))*100,
            "FT%":  float(compare_row.get("ft_pct",0))*100,
            "APG":  float(compare_row.get("ast_per_game",0)),
            "OREB": float(compare_row.get("oreb_per_game",0)),
            "eFG%": float(compare_row.get("efg_pct",0))*100,
            "TS%":  float(compare_row.get("ts_pct",0))*100,
            "Off Rtg":float(compare_row.get("off_rating",0)),
        }
        fig_off.add_trace(go.Bar(name=compare_team,
                                  x=list(cmp_off.keys()), y=list(cmp_off.values()),
                                  marker_color="#C9082A"))
    fig_off.update_layout(barmode="group", template="plotly_dark",
                           paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", height=380)
    st.plotly_chart(fig_off, use_container_width=True)
    st.dataframe(off_df.style.format({"Team": "{:.1f}", "League Avg": "{:.1f}", "vs League": "{:+.1f}"}),
                 use_container_width=True, hide_index=True)

with tab_def:
    def_metrics = {
        "Def Rtg":   float(team_row.get("def_rating", 0)),
        "SPG":       float(team_row.get("stl_per_game", 0)),
        "BPG":       float(team_row.get("blk_per_game", 0)),
        "DREB":      float(team_row.get("dreb_per_game", 0)),
        "Opp FG%":   float(team_row.get("opp_fg_pct", 0)) * 100,
        "Opp 3P%":   float(team_row.get("opp_fg3_pct", 0)) * 100,
        "TOV/G":     float(team_row.get("tov_per_game", 0)),
    }

    def_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in def_metrics.items()])
    fig_def = px.bar(def_df, x="Metric", y="Value", color="Value",
                      color_continuous_scale="Reds_r",
                      template="plotly_dark", title="Defensive Statistics")
    fig_def.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                           height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_def, use_container_width=True)

with tab_misc:
    adv = {
        "Net Rating":   float(team_row.get("net_rating", 0)),
        "Pace":         float(team_row.get("pace", 99)),
        "PIE":          float(team_row.get("pie", 0.5)) * 100,
        "TOV%":         float(team_row.get("tov_pct", 13)),
        "OREB%":        float(team_row.get("oreb_pct", 25)),
        "DREB%":        float(team_row.get("dreb_pct", 75)),
        "SRS":          float(team_row.get("srs", 0)),
        "Win%":         float(team_row.get("win_pct", 0.5)) * 100,
    }
    adv_df = pd.DataFrame([{"Metric": k, "Value": round(v, 2)} for k, v in adv.items()])
    st.dataframe(adv_df, use_container_width=True, hide_index=True)

# ── Player Breakdown ──────────────────────────────────────────────────────────
st.subheader("👤 Roster Player Breakdown")

if not player_stats.empty and "team_abbr" in player_stats.columns:
    from data.features import _team_abbr_from_name
    abbr = _team_abbr_from_name(selected_team)
    team_players = player_stats[
        player_stats["team_abbr"].str.upper() == abbr
    ].sort_values("pts_per_game", ascending=False)

    if not team_players.empty:
        display_cols = ["player_name", "min_per_game", "pts_per_game", "reb_per_game",
                         "ast_per_game", "stl_per_game", "blk_per_game",
                         "fg_pct", "fg3_pct", "ts_pct", "usg_pct", "net_rating"]
        available = [c for c in display_cols if c in team_players.columns]
        player_display = team_players[available].reset_index(drop=True)
        player_display.columns = [c.replace("_per_game","").replace("_"," ").upper()
                                    for c in player_display.columns]

        st.dataframe(
            player_display.style.background_gradient(
                subset=[c for c in player_display.columns if "PTS" in c or "NET" in c],
                cmap="Blues"
            ).format("{:.1f}", subset=[c for c in player_display.columns if c != "PLAYER NAME"]),
            use_container_width=True,
            hide_index=True,
        )

        # Scoring chart
        fig_players = px.bar(
            team_players.head(10),
            x="player_name", y="pts_per_game",
            color="usg_pct" if "usg_pct" in team_players else "pts_per_game",
            color_continuous_scale="Blues",
            labels={"pts_per_game": "PPG", "player_name": "Player", "usg_pct": "Usage%"},
            template="plotly_dark",
            title="Top 10 Scorers (colored by Usage Rate)",
        )
        fig_players.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                                    height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig_players, use_container_width=True)
    else:
        st.info("No player data for this team.")

# ── Injury Report for Team ────────────────────────────────────────────────────
st.subheader("🏥 Injury Report")
if not injury_df.empty and "team_name" in injury_df.columns:
    team_injuries = injury_df[
        injury_df["team_name"].str.lower().str.contains(
            selected_team.lower()[:8], na=False
        )
    ]
    if not team_injuries.empty:
        st.dataframe(
            team_injuries[["player_name", "position", "injury_type", "status", "updated"]]
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✅ No current injuries reported for this team.")
else:
    st.info("Injury data not available.")

# ── League Rankings ────────────────────────────────────────────────────────────
st.subheader("🏅 League Rankings")

if not team_stats.empty:
    rankings = {}
    for col, label, asc in [
        ("net_rating", "Net Rating", False),
        ("pts_per_game", "Scoring", False),
        ("def_rating", "Defense (lower=better)", True),
        ("off_rating", "Offense", False),
        ("efg_pct", "eFG%", False),
        ("pace", "Pace", False),
    ]:
        if col in team_stats.columns:
            ranked = team_stats.sort_values(col, ascending=asc).reset_index(drop=True)
            rank   = ranked[ranked["team_name"] == selected_team].index
            if len(rank) > 0:
                rankings[label] = f"#{rank[0]+1} / 30"

    rank_df = pd.DataFrame(list(rankings.items()), columns=["Category", "Rank"])
    st.dataframe(rank_df, use_container_width=True, hide_index=True)
