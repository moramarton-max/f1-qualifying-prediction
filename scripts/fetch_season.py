"""
Fetch all race weekends for a given season and save raw lap data to disk.

Usage:
    python scripts/fetch_season.py --year 2024
    python scripts/fetch_season.py --year 2024 --target Q --force
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fastf1

from src.data.fetcher import enable_cache, fetch_session
from src.data.loader import is_raw_available, save_raw
from src.data.session_config import get_preceding_sessions, is_sprint_weekend
from src.utils.logging_config import get_logger

logger = get_logger("fetch_season")


def main():
    parser = argparse.ArgumentParser(description="Fetch a full F1 season of practice/quali data.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--target", default="Q", choices=["Q", "SQ"])
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--raw-dir", default="data/raw/")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already on disk")
    args = parser.parse_args()

    enable_cache(args.cache_dir)

    schedule = fastf1.get_event_schedule(args.year, include_testing=False)
    rounds = schedule["RoundNumber"].tolist()

    logger.info(f"Season {args.year}: {len(rounds)} rounds found.")

    for round_number in rounds:
        event_name = schedule.loc[schedule["RoundNumber"] == round_number, "EventName"].values[0]
        sessions = get_preceding_sessions(args.year, round_number, args.target)

        for session in sessions:
            if not args.force and is_raw_available(args.year, round_number, session, args.raw_dir):
                logger.info(f"Already on disk: {args.year} R{round_number} ({event_name}) {session} — skip")
                continue

            logger.info(f"Fetching: {args.year} R{round_number} ({event_name}) {session}")
            laps = fetch_session(args.year, round_number, session, args.cache_dir)
            if laps is not None:
                save_raw(laps, args.year, round_number, session, args.raw_dir)


if __name__ == "__main__":
    main()
