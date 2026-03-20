"""
Fetch all race weekends for a given season and save raw lap data to disk.

Usage:
    python scripts/fetch_season.py --year 2024
    python scripts/fetch_season.py --year 2024 --target Q --delay 3
    python scripts/fetch_season.py --year 2024 --rounds 1 5 10
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fastf1

from src.data.fetcher import enable_cache, fetch_session
from src.data.loader import is_raw_available, save_raw
from src.data.session_config import get_preceding_sessions
from src.utils.logging_config import get_logger

logger = get_logger("fetch_season")


def fetch_quali_results(year: int, round_number: int, cache_dir: str = "cache/") -> None:
    """
    Fetch qualifying session results and save as data/raw/{year}_R{round}_Q_results.parquet.
    Only saves driver + final position — no laps needed, just session.results.
    """
    import pandas as pd
    results_path = f"data/raw/{year}_R{round_number:02d}_Q_results.parquet"
    if os.path.exists(results_path):
        return

    enable_cache(cache_dir)
    try:
        session = fastf1.get_session(year, round_number, "Qualifying")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results[["Abbreviation", "TeamName", "Position"]].copy()
        results.columns = ["Driver", "Team", "QualiPos"]
        results["QualiPos"] = results["QualiPos"].astype(int)
        results["Year"] = year
        results["Round"] = round_number
        os.makedirs("data/raw", exist_ok=True)
        results.to_parquet(results_path, index=False)
        logger.info(f"Saved quali results: {results_path}")
    except Exception as exc:
        logger.error(f"Failed to fetch quali results {year} R{round_number}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Fetch a full F1 season of practice/quali data.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--target", default="Q", choices=["Q", "SQ"])
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--raw-dir", default="data/raw/")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already on disk")
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between session fetches (default: 2). Increase if you hit rate limits."
    )
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=None,
        help="Fetch only specific round numbers (e.g. --rounds 1 2 3). Default: all rounds."
    )
    args = parser.parse_args()

    enable_cache(args.cache_dir)

    schedule = fastf1.get_event_schedule(args.year, include_testing=False)
    all_rounds = schedule["RoundNumber"].tolist()
    rounds = args.rounds if args.rounds else all_rounds

    logger.info(f"Season {args.year}: fetching {len(rounds)} of {len(all_rounds)} rounds.")

    for round_number in rounds:
        row = schedule.loc[schedule["RoundNumber"] == round_number]
        if row.empty:
            logger.warning(f"Round {round_number} not found in schedule — skipping")
            continue
        event_name = row["EventName"].values[0]
        sessions = get_preceding_sessions(args.year, round_number, args.target)

        for session in sessions:
            if not args.force and is_raw_available(args.year, round_number, session, args.raw_dir):
                logger.info(f"Skip (cached): {args.year} R{round_number} ({event_name}) {session}")
                continue

            logger.info(f"Fetching: {args.year} R{round_number} ({event_name}) {session}")
            laps = fetch_session(args.year, round_number, session, args.cache_dir)
            if laps is not None:
                save_raw(laps, args.year, round_number, session, args.raw_dir)

            time.sleep(args.delay)

        # Also fetch qualifying results (training labels) — very lightweight
        fetch_quali_results(args.year, round_number, args.cache_dir)
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
