"""
Odds Tracker Dashboard Page
Live betting lines, public money %, sharp money signals, and ATS records
pulled from Odds Shark and Action Network.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Odds Tracker | NBA Playoffs Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .odds-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border-radius: 12px; padding: 18px 22px; margin-bottom: 10px;
        border: 1px solid #2d3561;
    }
    .sharp-tag {
        background: #00c853; color: #000; border-radius: 6px;
        padding: 2px 8px; font-size: 11px; font-weight: 700;
    }
    .public-tag {
        background: #ff5252; color: #fff; border-radius: 6px;
        padding: 2px 8px; font-size: 11px; font-weight: 700;
    }
    .neutral-tag {
        background: #78909c; color: #fff; border-radius: 6px;
        padding: 2px 8px; font-size: 11px; font-weight: 700;
    }
    .edge-pos { color: #00e676; font-weight: 700; }
    .edge-neg { color: #ff5252; font-weight: 700; }
    .metric-box {
        background: #1a1f2e; border-radius: 8px; padding: 14px;
        text-align: center; border: 1px solid #2d3561;
    }
    .metric-val { font-size: 28px; font-weight: 800; color: #f7b731; }
    .metric-lbl { font-size: 12px; color: #90a4ae; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=180)
def load_odds_data():
    try:
        from data.odds_scraper import (
            get_odds_dataframe, get_all_current_odds,
            get_ats_records, get_action_network_odds,
        )
        df  = get_odds_dataframe()
        all_odds = get_all_current_odds()
        ats = get_ats_records()
        return df, all_odds, ats
    except Exception as e:
        st.warning(f"Live odds unavailable — showing demo data ({e})")
        return _demo_odds_df(), {}, _demo_ats()


def _demo_odds_df():
    return pd.DataFrame([
        {"Away": "MEM", "Home": "BOS", "Spread (Home)": -6.5, "Total (O/U)": 218.5,
         "Home ML": -220, "Away ML": 188, "Market Win% (Home)": 69.2,
         "Public Bets% (Home)": 62, "Public $% (Home)": 67, "Sharp Indicator": 5.0,
         "Line Movement": -1.0, "Home ATS%": 51.2, "Away ATS%": 48.7, "Source": "demo"},
        {"Away": "LAC", "Home": "OKC", "Spread (Home)": -5.5, "Total (O/U)": 222.0,
         "Home ML": -195, "Away ML": 165, "Market Win% (Home)": 66.1,
         "Public Bets% (Home)": 58, "Public $% (Home)": 63, "Sharp Indicator": 5.0,
         "Line Movement": 0.5, "Home ATS%": 53.1, "Away ATS%": 47.3, "Source": "demo"},
        {"Away": "PHX", "Home": "DEN", "Spread (Home)": -4.5, "Total (O/U)": 220.0,
         "Home ML": -175, "Away ML": 150, "Market Win% (Home)": 63.8,
         "Public Bets% (Home)": 55, "Public $% (Home)": 59, "Sharp Indicator": 4.0,
         "Line Movement": -0.5, "Home ATS%": 50.0, "Away ATS%": 49.3, "Source": "demo"},
        {"Away": "MIN", "Home": "NYK", "Spread (Home)": -2.5, "Total (O/U)": 221.0,
         "Home ML": -140, "Away ML": 120, "Market Win% (Home)": 58.5,
         "Public Bets% (Home)": 52, "Public $% (Home)": 55, "Sharp Indicator": 3.0,
         "Line Movement": 1.0, "Home ATS%": 49.2, "Away ATS%": 52.4, "Source": "demo"},
        {"Away": "MIA", "Home": "IND", "Spread (Home)": -1.5, "Total (O/U)": 223.5,
         "Home ML": -130, "Away ML": 110, "Market Win% (Home)": 56.5,
         "Public Bets% (Home)": 51, "Public $% (Home)": 53, "Sharp Indicator": 2.0,
         "Line Movement": 0.0, "Home ATS%": 48.9, "Away ATS%": 51.1, "Source": "demo"},
        {"Away": "DAL", "Home": "ORL", "Spread (Home)": -1.0, "Total (O/U)": 212.0,
         "Home ML": -120, "Away ML": 100, "Market Win% (Home)": 54.5,
         "Public Bets% (Home)": 50, "Public $% (Home)": 51, "Sharp Indicator": 1.0,
         "Line Movement": -0.5, "Home ATS%": 50.5, "Away ATS%": 50.0, "Source": "demo"},
        {"Away": "CHI", "Home": "DET", "Spread (Home)": 1.5, "Total (O/U)": 218.0,
         "Home ML": 100, "Away ML": -120, "Market Win% (Home)": 46.5,
         "Public Bets% (Home)": 48, "Public $% (Home)": 46, "Sharp Indicator": -2.0,
         "Line Movement": 1.5, "Home ATS%": 51.3, "Away ATS%": 48.6, "Source": "demo"},
        {"Away": "SAC", "Home": "GSW", "Spread (Home)": -3.0, "Total (O/U)": 225.5,
         "Home ML": -155, "Away ML": 132, "Market Win% (Home)": 61.0,
         "Public Bets% (Home)": 57, "Public $% (Home)": 60, "Sharp Indicator": 3.0,
         "Line Movement": -1.5, "Home ATS%": 49.8, "Away ATS%": 50.2, "Source": "demo"},
    ])


def _demo_ats():
    from config import NBA_TEAMS
    rng = np.random.default_rng(99)
    return {abbr: {"ats_pct": round(float(rng.normal(0.5, 0.04)), 4),
                   "ou_over_pct": round(float(rng.normal(0.5, 0.035)), 4),
                   "ats_wins": 28, "ats_losses": 24}
            for abbr in NBA_TEAMS}


# ── Header ─────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📈 Live Odds Tracker")
    st.caption("Betting lines · Public money · Sharp signals · ATS records — Odds Shark & Action Network")
with col_h2:
    if st.button("🔄 Refresh Odds", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Last update: {datetime.now().strftime('%I:%M:%S %p')}")

st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Fetching live odds..."):
    odds_df, all_odds, ats_records = load_odds_data()

if odds_df.empty:
    st.warning("No odds data found. The NBA off-season may be active or sources are temporarily unavailable.")
    st.stop()

# ── Summary Metrics Row ────────────────────────────────────────────────────────
n_games     = len(odds_df)
avg_total   = odds_df["Total (O/U)"].mean() if "Total (O/U)" in odds_df else 220
sharp_games = (odds_df["Sharp Indicator"].abs() >= 5).sum() if "Sharp Indicator" in odds_df else 0
n_fav_home  = (odds_df["Spread (Home)"] < 0).sum() if "Spread (Home)" in odds_df else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{n_games}</div>
        <div class="metric-lbl">Games with Lines</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{avg_total:.1f}</div>
        <div class="metric-lbl">Avg Over/Under</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{sharp_games}</div>
        <div class="metric-lbl">Sharp Money Spots</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{n_fav_home}</div>
        <div class="metric-lbl">Home Favorites</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Game-by-Game Cards ────────────────────────────────────────────────────────
st.subheader("🎯 Today's Matchups — Lines & Public Action")

for _, row in odds_df.iterrows():
    away     = row.get("Away", "??")
    home     = row.get("Home", "??")
    spread   = row.get("Spread (Home)", 0)
    total    = row.get("Total (O/U)", 220)
    home_ml  = row.get("Home ML", -110)
    away_ml  = row.get("Away ML", -110)
    mkt_wp   = row.get("Market Win% (Home)", 50)
    bet_pct  = row.get("Public Bets% (Home)", 50)
    mon_pct  = row.get("Public $% (Home)", 50)
    sharp    = row.get("Sharp Indicator", 0)
    movement = row.get("Line Movement", 0)
    home_ats = row.get("Home ATS%", 50)
    away_ats = row.get("Away ATS%", 50)

    spread_str = f"{spread:+.1f}" if spread != 0 else "PK"
    ml_home_str = f"+{int(home_ml)}" if home_ml > 0 else str(int(home_ml))
    ml_away_str = f"+{int(away_ml)}" if away_ml > 0 else str(int(away_ml))

    # Sharp money direction label
    if sharp >= 5:
        sharp_html = f'<span class="sharp-tag">⚡ SHARP → {home}</span>'
    elif sharp <= -5:
        sharp_html = f'<span class="sharp-tag">⚡ SHARP → {away}</span>'
    else:
        sharp_html = f'<span class="neutral-tag">📊 Balanced</span>'

    # Movement label
    if movement < -0.5:
        move_html = f'<span style="color:#00e676">▼ {movement:+.1f} (home gaining)</span>'
    elif movement > 0.5:
        move_html = f'<span style="color:#ff5252">▲ +{movement:.1f} (away gaining)</span>'
    else:
        move_html = f'<span style="color:#90a4ae">→ Stable</span>'

    st.markdown(f"""
    <div class="odds-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:18px; font-weight:700; color:#f0f0f0;">
                {away} <span style="color:#90a4ae; font-size:14px;">@</span> {home}
            </div>
            <div>{sharp_html}</div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-top:14px;">
            <div style="text-align:center;">
                <div style="color:#f7b731; font-size:20px; font-weight:800;">{spread_str}</div>
                <div style="color:#90a4ae; font-size:11px;">Spread (Home)</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#f7b731; font-size:20px; font-weight:800;">{total:.1f}</div>
                <div style="color:#90a4ae; font-size:11px;">Over/Under</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#80cbc4; font-size:16px; font-weight:700;">{ml_home_str} / {ml_away_str}</div>
                <div style="color:#90a4ae; font-size:11px;">Moneyline H / A</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#ce93d8; font-size:16px; font-weight:700;">{mkt_wp:.1f}%</div>
                <div style="color:#90a4ae; font-size:11px;">Mkt Win% (Home)</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#ef9a9a; font-size:14px;">{move_html}</div>
                <div style="color:#90a4ae; font-size:11px;">Line Movement</div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:10px; padding-top:10px; border-top:1px solid #2d3561;">
            <div>
                <span style="color:#90a4ae; font-size:12px;">Public Bets: </span>
                <span style="color:#fff; font-size:13px; font-weight:600;">{home}: {bet_pct:.0f}% | {away}: {100-bet_pct:.0f}%</span>
            </div>
            <div>
                <span style="color:#90a4ae; font-size:12px;">Public Money: </span>
                <span style="color:#fff; font-size:13px; font-weight:600;">{home}: {mon_pct:.0f}% | {away}: {100-mon_pct:.0f}%</span>
            </div>
            <div>
                <span style="color:#90a4ae; font-size:12px;">ATS Record: </span>
                <span style="color:#fff; font-size:13px; font-weight:600;">{home}: {home_ats:.1f}% | {away}: {away_ats:.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Charts Section ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Public vs Sharp Money",
    "📉 Line Movement",
    "🏆 ATS Standings",
    "🎲 O/U Analysis",
])

with tab1:
    st.subheader("Public Bets% vs Public Money% — Spot Sharp Action")
    st.caption("When money% >> bets%, sharp/professional bettors are on that side.")

    matchup_labels = [f"{r['Away']}@{r['Home']}" for _, r in odds_df.iterrows()]
    bet_pcts   = odds_df["Public Bets% (Home)"].tolist()
    money_pcts = odds_df["Public $% (Home)"].tolist()

    fig_sharp = go.Figure()
    fig_sharp.add_trace(go.Bar(
        name="Public Bets % (Home)",
        x=matchup_labels,
        y=bet_pcts,
        marker_color="#17408B",
        opacity=0.85,
    ))
    fig_sharp.add_trace(go.Bar(
        name="Public Money % (Home)",
        x=matchup_labels,
        y=money_pcts,
        marker_color="#F7B731",
        opacity=0.85,
    ))
    fig_sharp.add_hline(y=50, line_dash="dash", line_color="#90a4ae",
                        annotation_text="50% line")
    fig_sharp.update_layout(
        barmode="group",
        template="plotly_dark",
        height=380,
        legend=dict(orientation="h", y=1.1),
        yaxis_title="Percentage (%)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    st.plotly_chart(fig_sharp, use_container_width=True)

    # Sharp indicator table
    st.subheader("Sharp Money Indicator (Money% − Bets%)")
    st.caption("Positive = sharp money on home team · Negative = sharp money on away team")
    sharp_df = pd.DataFrame({
        "Matchup": matchup_labels,
        "Sharp Indicator": odds_df["Sharp Indicator"].tolist(),
        "Signal": ["⚡ Sharp Home" if s >= 5 else ("⚡ Sharp Away" if s <= -5 else "📊 Balanced")
                   for s in odds_df["Sharp Indicator"].tolist()],
    })
    st.dataframe(
        sharp_df.style.background_gradient(subset=["Sharp Indicator"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )


with tab2:
    st.subheader("Line Movement — Opening vs Current Spread")
    st.caption("Positive = line moved in favor of away team · Negative = moved toward home")

    fig_move = go.Figure(go.Bar(
        x=matchup_labels,
        y=odds_df["Line Movement"].tolist(),
        marker_color=["#00e676" if v < 0 else "#ff5252"
                      for v in odds_df["Line Movement"].tolist()],
        text=[f"{v:+.1f}" for v in odds_df["Line Movement"].tolist()],
        textposition="outside",
    ))
    fig_move.add_hline(y=0, line_color="#90a4ae", line_dash="dash")
    fig_move.update_layout(
        template="plotly_dark", height=360,
        yaxis_title="Spread Change (pts)",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    )
    st.plotly_chart(fig_move, use_container_width=True)

    # Current vs implied
    st.subheader("Market Implied Win Probability (Vig-Removed)")
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        name="Home Win Prob",
        x=matchup_labels,
        y=odds_df["Market Win% (Home)"].tolist(),
        marker_color="#17408B",
    ))
    fig_prob.add_trace(go.Bar(
        name="Away Win Prob",
        x=matchup_labels,
        y=[100 - v for v in odds_df["Market Win% (Home)"].tolist()],
        marker_color="#C9082A",
    ))
    fig_prob.add_hline(y=50, line_dash="dash", line_color="#f7b731")
    fig_prob.update_layout(
        barmode="stack", template="plotly_dark", height=360,
        yaxis_title="Win Probability (%)", yaxis_range=[0, 100],
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    )
    st.plotly_chart(fig_prob, use_container_width=True)


with tab3:
    st.subheader("Against The Spread (ATS) Season Records")
    st.caption("Teams covering the spread more than 52% are value plays. Data from Odds Shark.")

    if ats_records:
        ats_rows = []
        for abbr, rec in ats_records.items():
            ats_rows.append({
                "Team":      abbr,
                "ATS W":     rec.get("ats_wins", 0),
                "ATS L":     rec.get("ats_losses", 0),
                "ATS P":     rec.get("ats_pushes", 0),
                "ATS%":      round(rec.get("ats_pct", 0.5) * 100, 1),
                "O/U Over%": round(rec.get("ou_over_pct", 0.5) * 100, 1),
            })
        ats_df = pd.DataFrame(ats_rows).sort_values("ATS%", ascending=False)

        # Color-coded bar chart
        fig_ats = go.Figure(go.Bar(
            x=ats_df["Team"],
            y=ats_df["ATS%"],
            marker_color=["#00e676" if v > 52 else ("#ff5252" if v < 48 else "#f7b731")
                          for v in ats_df["ATS%"]],
            text=[f"{v:.1f}%" for v in ats_df["ATS%"]],
            textposition="outside",
        ))
        fig_ats.add_hline(y=50, line_dash="dash", line_color="#90a4ae",
                          annotation_text="50% baseline")
        fig_ats.update_layout(
            template="plotly_dark", height=420,
            yaxis_title="ATS Win %", yaxis_range=[35, 65],
            xaxis_tickangle=-45,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_ats, use_container_width=True)

        st.dataframe(
            ats_df.style
                .background_gradient(subset=["ATS%"], cmap="RdYlGn", vmin=42, vmax=58)
                .background_gradient(subset=["O/U Over%"], cmap="YlOrRd", vmin=42, vmax=58),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("ATS records loading...")


with tab4:
    st.subheader("Over/Under Totals Analysis")

    col_ou1, col_ou2 = st.columns(2)

    with col_ou1:
        # Totals by game
        fig_tot = go.Figure(go.Bar(
            x=matchup_labels,
            y=odds_df["Total (O/U)"].tolist(),
            marker_color="#ce93d8",
            text=[f"{v:.1f}" for v in odds_df["Total (O/U)"].tolist()],
            textposition="outside",
        ))
        avg_t = odds_df["Total (O/U)"].mean()
        fig_tot.add_hline(y=avg_t, line_dash="dash", line_color="#f7b731",
                          annotation_text=f"Avg: {avg_t:.1f}")
        fig_tot.update_layout(
            title="Game Totals (O/U)",
            template="plotly_dark", height=350,
            yaxis_title="Points", yaxis_range=[195, 245],
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_tot, use_container_width=True)

    with col_ou2:
        if ats_records:
            # Team tendency to go over
            ou_rows = [(abbr, round(rec.get("ou_over_pct", 0.5) * 100, 1))
                       for abbr, rec in ats_records.items()]
            ou_df = pd.DataFrame(ou_rows, columns=["Team", "Over%"]).sort_values("Over%", ascending=False)

            fig_ou_team = go.Figure(go.Bar(
                x=ou_df.head(15)["Team"],
                y=ou_df.head(15)["Over%"],
                marker_color=["#00e676" if v > 52 else "#ef9a9a"
                              for v in ou_df.head(15)["Over%"]],
                text=[f"{v:.1f}%" for v in ou_df.head(15)["Over%"]],
                textposition="outside",
            ))
            fig_ou_team.add_hline(y=50, line_dash="dash", line_color="#90a4ae")
            fig_ou_team.update_layout(
                title="Team O/U Over % (Top 15)",
                template="plotly_dark", height=350,
                yaxis_title="Over %", yaxis_range=[40, 62],
                xaxis_tickangle=-45,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            )
            st.plotly_chart(fig_ou_team, use_container_width=True)

st.divider()

# ── Raw Odds Table ─────────────────────────────────────────────────────────────
with st.expander("📋 Full Odds Data Table", expanded=False):
    display_cols = [c for c in odds_df.columns if c != "Source"]
    styled = odds_df[display_cols].style.background_gradient(
        subset=["Market Win% (Home)"], cmap="RdYlGn", vmin=35, vmax=65
    ).background_gradient(
        subset=["Sharp Indicator"], cmap="RdYlGn", vmin=-15, vmax=15
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        f"Source: Odds Shark + Action Network · "
        f"{'⚠️ Demo data (live scraping unavailable)' if (not odds_df.empty and 'demo' in odds_df['Source'].values) else '✅ Live data'}"
    )

# ── Model vs Market Edge ───────────────────────────────────────────────────────
st.subheader("🔍 Model vs. Market Edge")
st.caption(
    "The ML model probability vs. the market-implied win probability. "
    "A large positive edge on a team suggests the model sees more value than the betting market."
)

try:
    from models.predictor import GamePredictor
    from data.features import build_matchup_features

    predictor = GamePredictor()
    if predictor.load():
        edge_rows = []
        for _, row in odds_df.iterrows():
            try:
                away_team = row["Away"]
                home_team = row["Home"]
                feat = build_matchup_features(home_team, away_team)
                pred = predictor.predict_game(feat)
                edge = pred.get("market_edge", 0.0)
                edge_rows.append({
                    "Matchup":        f"{away_team} @ {home_team}",
                    "Model Win% (Home)": round(pred["home_win_prob"] * 100, 1),
                    "Market Win% (Home)": row["Market Win% (Home)"],
                    "Edge":           round(edge * 100, 1),
                    "Signal":         "✅ Lean Home" if edge > 0.05 else ("🔴 Lean Away" if edge < -0.05 else "➖ Neutral"),
                })
            except Exception:
                continue

        if edge_rows:
            edge_df = pd.DataFrame(edge_rows).sort_values("Edge", ascending=False)
            st.dataframe(
                edge_df.style.background_gradient(subset=["Edge"], cmap="RdYlGn", vmin=-15, vmax=15),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Train the model first (`python scripts/train_model.py`) to see model vs market edges.")
    else:
        st.info("Model not trained yet. Run `python scripts/train_model.py` to generate predictions.")
except Exception as e:
    st.info(f"Model predictions unavailable: {e}")

st.markdown("---")
st.caption(
    "📈 **Data Sources:** Odds Shark (odds, ATS records) · Action Network (public betting %, line movement)  \n"
    "⚠️ *Odds data is for informational and predictive modeling purposes only. This tool does not facilitate betting.*"
)
