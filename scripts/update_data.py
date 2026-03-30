"""
Real-Time Data Update Script
Runs continuously, fetching the latest NBA data every N minutes.
Automatically retriggers predictions when:
  - A new game result is recorded
  - An injury status changes
  - A player is added to or cleared from the injury report

Usage:
    python scripts/update_data.py              # Run once
    python scripts/update_data.py --loop       # Run continuously every 5 min
    python scripts/update_data.py --loop --interval 120  # Every 2 min
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import CACHE_TTL_SECONDS, INJURY_REFRESH_SECONDS
from data.nba_data import (
    get_team_stats, get_player_stats, get_live_scores, get_standings
)
from data.injury_tracker import (
    get_injury_report, get_team_injury_summary, watch_for_injury_changes
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RealTimeUpdater:
    """Manages real-time NBA data fetching and change detection."""

    def __init__(self):
        self.prev_injury_df    = None
        self.prev_scores       = []
        self.update_count      = 0
        self.changes_detected  = 0

    def run_once(self) -> dict:
        """
        Perform one full update cycle.
        Returns a summary of what changed.
        """
        now = datetime.now().strftime("%I:%M:%S %p")
        logger.info(f"━━━ Update cycle {self.update_count + 1} | {now} ━━━")

        changes = {"injuries": [], "scores": [], "timestamp": now}

        # ── Injury Report ──────────────────────────────────────────────────────
        try:
            logger.info("  Checking injury report...")
            current_inj = get_injury_report(force_refresh=True)
            if not current_inj.empty:
                logger.info(f"  Injury report: {len(current_inj)} players listed")

                # Detect changes
                if self.prev_injury_df is not None:
                    inj_changes = watch_for_injury_changes(self.prev_injury_df)
                    if inj_changes:
                        for ch in inj_changes:
                            self.changes_detected += 1
                            if ch["type"] == "new_injury":
                                msg = (f"🚨 NEW INJURY: {ch['player']} ({ch['team']}) "
                                       f"— {ch['injury']} | Status: {ch['status']}")
                                logger.warning(msg)
                            elif ch["type"] == "cleared":
                                msg = (f"✅ CLEARED: {ch['player']} ({ch['team']}) "
                                       f"is no longer on injury report")
                                logger.info(msg)
                            elif ch["type"] == "status_change":
                                msg = (f"🔄 STATUS CHANGE: {ch['player']} ({ch['team']}) "
                                       f"{ch['old_status']} → {ch['new_status']}")
                                logger.info(msg)
                            changes["injuries"].append(ch)

                self.prev_injury_df = current_inj
            else:
                logger.warning("  Injury report returned empty.")
        except Exception as e:
            logger.error(f"  Injury update failed: {e}")

        # ── Team Injury Summary ────────────────────────────────────────────────
        try:
            logger.info("  Updating team injury summaries...")
            summary = get_team_injury_summary(force_refresh=True)
            high_impact = {k: v for k, v in summary.items()
                           if v.get("impact_score", 0) > 0.3}
            if high_impact:
                logger.warning(f"  ⚠️ {len(high_impact)} teams with significant injuries:")
                for team, data in high_impact.items():
                    logger.warning(f"    {team}: {data['severity']} ({data['impact_score']:.0%} impact)")
        except Exception as e:
            logger.error(f"  Team injury summary failed: {e}")

        # ── Live Scores ────────────────────────────────────────────────────────
        try:
            logger.info("  Fetching live scores...")
            live = get_live_scores()
            if live:
                logger.info(f"  Found {len(live)} game(s):")
                for g in live:
                    score_str = (f"    {g.get('away_team','')} {g.get('away_score',0)} @ "
                                 f"{g.get('home_team','')} {g.get('home_score',0)} "
                                 f"({g.get('status','')})")
                    logger.info(score_str)

                # Detect completed games
                for g in live:
                    if "Final" in str(g.get("status", "")):
                        game_id = g.get("game_id", "")
                        prev_ids = [pg.get("game_id") for pg in self.prev_scores]
                        if game_id not in prev_ids:
                            winner = (g["home_team"] if int(g.get("home_score", 0)) >
                                      int(g.get("away_score", 0)) else g["away_team"])
                            logger.info(f"  🏁 GAME FINAL: {winner} wins!")
                            changes["scores"].append({"game": g, "winner": winner})

                self.prev_scores = live
            else:
                logger.info("  No games today or scores unavailable.")
        except Exception as e:
            logger.error(f"  Live score update failed: {e}")

        # ── Team Stats (periodic refresh) ──────────────────────────────────────
        if self.update_count % 12 == 0:  # Every hour (if running every 5 min)
            try:
                logger.info("  Refreshing team stats (hourly)...")
                get_team_stats(force_refresh=True)
                get_player_stats(force_refresh=True)
                logger.info("  Team stats refreshed.")
            except Exception as e:
                logger.error(f"  Team stats refresh failed: {e}")

        self.update_count += 1
        return changes

    def loop(self, interval_seconds: int = 300) -> None:
        """Run the updater in a continuous loop."""
        logger.info(f"🏀 NBA Real-Time Updater started. Interval: {interval_seconds}s")
        logger.info("Press Ctrl+C to stop.\n")

        while True:
            try:
                changes = self.run_once()

                if changes["injuries"] or changes["scores"]:
                    logger.info(f"  → {len(changes['injuries'])} injury change(s), "
                                f"{len(changes['scores'])} score event(s) detected")
                    logger.info("  → Predictions will be updated on next dashboard load.")
                else:
                    logger.info("  → No changes detected.")

                logger.info(f"  Next update in {interval_seconds}s...\n")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("\n✋ Updater stopped.")
                logger.info(f"Total updates: {self.update_count} | Changes: {self.changes_detected}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in update loop: {e}")
                logger.info("Retrying in 30s...")
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="NBA Real-Time Data Updater")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=300,
                        help="Update interval in seconds (default: 300 = 5 minutes)")
    args = parser.parse_args()

    updater = RealTimeUpdater()

    if args.loop:
        updater.loop(interval_seconds=args.interval)
    else:
        changes = updater.run_once()
        logger.info(f"\nUpdate complete. Changes: {changes}")


if __name__ == "__main__":
    main()
