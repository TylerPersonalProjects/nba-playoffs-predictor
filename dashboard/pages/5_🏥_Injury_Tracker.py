"""
Injury Tracker Page
Real-time injury updates, team impact scores, and prediction adjustment tracker.
Auto-refreshes every 3 minutes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Injury Tracker | NBA Predictor", page_icon="🏥", layout="wide")

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=180)  # 3-minute cache
def load_injury_data():
    from data.injury_tracker import get_injury_report, get_team_injury_summary
    from data.nba_data import get_player_stats
    ir  = get_injury_report()
    ts  = get_team_injury_summary()
    ps  = get_player_stats()
    return ir, ts, ps

@st.cache_data(ttl=180)
def load_team_stats():
    from data.nba_data import get_team_stats
    return get_team_stats()

st.title("🏥 Real-Time Injury Tracker")
st.caption("Auto-refreshes every 3 minutes · Source: ESPN Injury Reports")

# Refresh button
col_ref, col_time = st.columns([1, 4])
with col_ref:
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()
with col_time:
    st.caption(f"Last checked: {datetime.now().strftime('%I:%M:%S %p')} · {datetime.now().strftime('%B %d, %Y')}")

st.markdown("---")

try:
    injury_df, team_inj_summary, player_stats = load_injury_data()
    team_stats = load_team_stats()
    data_ok = True
except Exception as e:
    st.error(f"Error loading injury data: {e}")
    data_ok = False
    injury_df = pd.DataFrame()
    team_inj_summary = {}
    player_stats = pd.DataFrame()
    team_stats = pd.DataFrame()

# ── Summary Metrics ───────────────────────────────────────────────────────────
if data_ok and not injury_df.empty:
    total_injured = len(injury_df)
    out_count = len(injury_df[injury_df.get("status","").str.lower().str.contains("out", na=False)]) \
        if "status" in injury_df.columns else 0
    doubt_count = len(injury_df[injury_df.get("status","").str.lower().str.contains("doubtful|questionable", na=False)]) \
        if "status" in injury_df.columns else 0
    teams_affected = injury_df["team_name"].nunique() if "team_name" in injury_df.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Injured", total_injured, delta=None)
    m2.metric("Confirmed Out", out_count)
    m3.metric("Doubtful/Questionable", doubt_count)
    m4.metric("Teams Affected", teams_affected)
else:
    st.warning("⚠️ Injury data could not be loaded from ESPN. Showing fallback data.")
    # Create minimal fallback
    injury_df = pd.DataFrame([
        {"player_name": "Joel Embiid",    "team_name": "Philadelphia 76ers", "position": "C",
         "injury_type": "Knee",  "status": "Out",         "status_weight": 1.0},
        {"player_name": "Kawhi Leonard",  "team_name": "LA Clippers",        "position": "SF",
         "injury_type": "Knee",  "status": "Out",         "status_weight": 1.0},
        {"player_name": "Klay Thompson",  "team_name": "Dallas Mavericks",   "position": "SG",
         "injury_type": "Ankle", "status": "Questionable","status_weight": 0.4},
    ])
    team_inj_summary = {
        "Philadelphia 76ers": {"impact_score": 0.75, "pts_lost_per_game": 22.0,
                                "injured_players": ["Joel Embiid (Out)"], "severity": "Critical"},
        "LA Clippers": {"impact_score": 0.60, "pts_lost_per_game": 18.0,
                         "injured_players": ["Kawhi Leonard (Out)"], "severity": "Critical"},
    }

st.markdown("---")

# ── Team Impact Chart ─────────────────────────────────────────────────────────
st.subheader("💥 Team Injury Impact Scores")

if team_inj_summary:
    impact_rows = []
    for team, data in team_inj_summary.items():
        impact_rows.append({
            "Team": team,
            "Impact Score": round(data.get("impact_score", 0) * 100, 1),
            "Pts Lost/Game": data.get("pts_lost_per_game", 0),
            "Severity": data.get("severity", "None"),
            "Injured Players": ", ".join(data.get("injured_players", [])[:3]),
        })

    impact_df = pd.DataFrame(impact_rows).sort_values("Impact Score", ascending=False)

    if not impact_df.empty:
        color_map = {
            "Critical": "#ff4444",
            "High":     "#ff8c00",
            "Moderate": "#F7B731",
            "Low":      "#88ccff",
            "None":     "#44ff88",
        }
        impact_df["Color"] = impact_df["Severity"].map(color_map).fillna("#aaa")

        fig_impact = go.Figure()
        fig_impact.add_trace(go.Bar(
            x=impact_df["Team"],
            y=impact_df["Impact Score"],
            marker_color=impact_df["Color"],
            text=[f"{s} ({sc:.0f}%)" for s, sc in zip(impact_df["Severity"], impact_df["Impact Score"])],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Impact: %{y:.1f}%<br>%{customdata}<extra></extra>",
            customdata=impact_df["Injured Players"],
        ))
        fig_impact.update_layout(
            title="Injury Impact by Team (higher = more impacted)",
            xaxis_tickangle=-35,
            yaxis_title="Impact Score (%)",
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_impact, use_container_width=True)

        # Severity legend
        st.markdown("""
        <div style="display:flex; gap:1.5rem; margin:0.5rem 0">
            <span style="color:#ff4444">■ Critical (&gt;60% impact)</span>
            <span style="color:#ff8c00">■ High (35-60%)</span>
            <span style="color:#F7B731">■ Moderate (15-35%)</span>
            <span style="color:#88ccff">■ Low (&lt;15%)</span>
            <span style="color:#44ff88">■ None</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Full Injury Report Table ──────────────────────────────────────────────────
st.subheader("📋 Full Injury Report")

# Filters
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    status_filter = st.multiselect(
        "Status Filter",
        options=["Out", "Doubtful", "Questionable", "Day-To-Day", "Probable"],
        default=["Out", "Doubtful", "Questionable"],
    )
with col_f2:
    if "team_name" in injury_df.columns:
        team_filter = st.selectbox("Team", ["All Teams"] + sorted(injury_df["team_name"].dropna().unique().tolist()))
    else:
        team_filter = "All Teams"
with col_f3:
    if "injury_type" in injury_df.columns:
        injury_filter = st.multiselect("Injury Type", options=sorted(injury_df["injury_type"].dropna().unique().tolist()),
                                       default=[])
    else:
        injury_filter = []

# Apply filters
disp_df = injury_df.copy()
if status_filter and "status" in disp_df.columns:
    disp_df = disp_df[disp_df["status"].isin(status_filter)]
if team_filter != "All Teams" and "team_name" in disp_df.columns:
    disp_df = disp_df[disp_df["team_name"] == team_filter]
if injury_filter and "injury_type" in disp_df.columns:
    disp_df = disp_df[disp_df["injury_type"].isin(injury_filter)]

if not disp_df.empty:
    # Style the table
    show_cols = [c for c in ["player_name","team_name","position","injury_type","status","updated"]
                  if c in disp_df.columns]
    disp_df_show = disp_df[show_cols].reset_index(drop=True)
    disp_df_show.columns = [c.replace("_"," ").title() for c in disp_df_show.columns]

    def color_status(row):
        status = row.get("Status", "")
        if "Out" in str(status):
            return ["color: #ff4444; font-weight: 700"] * len(row)
        if "Doubtful" in str(status):
            return ["color: #ff8c00; font-weight: 600"] * len(row)
        if "Questionable" in str(status):
            return ["color: #F7B731"] * len(row)
        return ["color: #44ff88"] * len(row)

    st.dataframe(
        disp_df_show.style.apply(color_status, axis=1),
        use_container_width=True,
        hide_index=True,
        height=400,
    )
    st.caption(f"{len(disp_df_show)} players matching filters")
else:
    st.success("✅ No injuries matching your filters.")

st.markdown("---")

# ── Injury Type Breakdown ─────────────────────────────────────────────────────
st.subheader("🔍 Injury Type Breakdown")
if not injury_df.empty and "injury_type" in injury_df.columns:
    inj_type_counts = injury_df["injury_type"].value_counts().head(10).reset_index()
    inj_type_counts.columns = ["Injury Type", "Count"]

    fig_types = px.pie(
        inj_type_counts,
        names="Injury Type",
        values="Count",
        hole=0.4,
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Plasma,
        title="Most Common Injury Types This Season",
    )
    fig_types.update_layout(paper_bgcolor="#1a1a2e", height=380)

    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.plotly_chart(fig_types, use_container_width=True)

    with col_bar:
        # Status breakdown
        if "status" in injury_df.columns:
            status_counts = injury_df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            status_color_map = {
                "Out": "#ff4444", "Doubtful": "#ff8c00",
                "Questionable": "#F7B731", "Day-To-Day": "#88ccff",
                "Probable": "#44ff88",
            }
            fig_status = px.bar(
                status_counts,
                x="Status", y="Count",
                color="Status",
                color_discrete_map=status_color_map,
                template="plotly_dark",
                title="Players by Injury Status",
            )
            fig_status.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                                      height=380, showlegend=False)
            st.plotly_chart(fig_status, use_container_width=True)

st.markdown("---")

# ── Prediction Impact Panel ───────────────────────────────────────────────────
st.subheader("🔮 How Injuries Are Affecting Predictions")
st.caption("Showing teams where injuries have the biggest impact on playoff win probability")

if team_inj_summary:
    critical_teams = {k: v for k, v in team_inj_summary.items()
                       if v.get("impact_score", 0) > 0.3}
    if critical_teams:
        for team, data in list(critical_teams.items())[:5]:
            sev   = data.get("severity", "None")
            score = data.get("impact_score", 0)
            pts   = data.get("pts_lost_per_game", 0)
            names = data.get("injured_players", [])
            color = {"Critical": "#ff4444", "High": "#ff8c00", "Moderate": "#F7B731"}.get(sev, "#aaa")

            st.markdown(f"""
            <div style="background:#1a1a2e; border-left:5px solid {color}; border-radius:10px;
                        padding:1rem; margin:0.5rem 0">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <div>
                        <span style="font-weight:800; color:#fff; font-size:1.1rem">{team}</span>
                        <span style="background:{color}; color:#000; padding:2px 8px;
                                     border-radius:10px; font-size:0.75rem; margin-left:8px;
                                     font-weight:700">{sev}</span>
                    </div>
                    <div style="text-align:right">
                        <span style="color:{color}; font-size:1.5rem; font-weight:900">-{pts:.0f} PPG</span>
                        <div style="color:#aaa; font-size:0.75rem">estimated points lost</div>
                    </div>
                </div>
                <div style="color:#aaa; font-size:0.85rem; margin-top:0.5rem">
                    {" · ".join(names[:4])}
                </div>
                <div style="background:#222; border-radius:6px; height:8px; margin-top:8px; overflow:hidden">
                    <div style="background:{color}; width:{min(score*100, 100):.0f}%;
                                height:100%; border-radius:6px"></div>
                </div>
                <div style="color:#666; font-size:0.75rem; margin-top:4px">
                    Impact score: {score:.0%} (affects predictions by ~{score*15:.0f}% win probability)
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No teams are currently critically impacted by injuries.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🔄 This page auto-refreshes every 3 minutes to show the latest injury news.")
st.markdown('<meta http-equiv="refresh" content="180">', unsafe_allow_html=True)
