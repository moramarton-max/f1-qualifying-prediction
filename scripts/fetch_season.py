"""
Fetch all race weekends for a given season and save raw lap data to disk.

Uses ThreadPoolExecutor to fetch multiple rounds in parallel.
Each worker handles one round sequentially (FP1 → FP2 → FP3 → Q results)
so per-round ordering is preserved and API requests are naturally spread out.

Usage:
    python scripts/fetch_season.py --year 2024
    python scripts/fetch_season.py --year 2024 --workers 4 --delay 1
    python scripts/fetch_season.py --year 2024 --rounds 1 5 10
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fastf1

from src.data.fetcher import enable_cache, fetch_session
from src.data.loader import is_raw_available, save_raw
from src.data.session_config import get_preceding_sessions
from src.utils.logging_config import get_logger

logger = get_logger("fetch_season")


def fetch_quali_results(year: int, round_number: int, cache_dir: str = "cache/") -> None:
    """Fetch qualifying results (training labels). Very lightweight — no laps loaded."""
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


def fetch_round(
    year: int,
    round_number: int,
    event_name: str,
    target: str,
    cache_dir: str,
    raw_dir: str,
    delay: float,
    force: bool,
) -> list[str]:
    """
    Fetch all sessions for one round. Runs inside a thread.
    Returns list of saved file paths.
    """
    saved = []
    sessions = get_preceding_sessions(year, round_number, target)

    for session in sessions:
        if not force and is_raw_available(year, round_number, session, raw_dir):
            logger.info(f"Skip (cached): {year} R{round_number} ({event_name}) {session}")
            continue

        logger.info(f"Fetching: {year} R{round_number} ({event_name}) {session}")
        laps = fetch_session(year, round_number, session, cache_dir)
        if laps is not None:
            save_raw(laps, year, round_number, session, raw_dir)
            saved.append(f"{year}_R{round_number:02d}_{session}")
        time.sleep(delay)

    fetch_quali_results(year, round_number, cache_dir)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--target", default="Q", choices=["Q", "SQ"])
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--raw-dir", default="data/raw/")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel round fetchers (default: 3). Don't go above 5 — API rate limits.",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between session fetches within a worker (default: 1).",
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    enable_cache(args.cache_dir)

    schedule = fastf1.get_event_schedule(args.year, include_testing=False)
    all_rounds = schedule["RoundNumber"].tolist()
    rounds = args.rounds if args.rounds else all_rounds

    logger.info(
        f"Season {args.year}: {len(rounds)} rounds, "
        f"{args.workers} workers, {args.delay}s delay."
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_round,
                args.year,
                rnd,
                schedule.loc[schedule["RoundNumber"] == rnd, "EventName"].values[0],
                args.target,
                args.cache_dir,
                args.raw_dir,
                args.delay,
                args.force,
            ): rnd
            for rnd in rounds
        }

        total_saved = 0
        for future in as_completed(futures):
            rnd = futures[future]
            try:
                saved = future.result()
                total_saved += len(saved)
                logger.info(f"Round {rnd} done — {len(saved)} new files saved.")
            except Exception as exc:
                logger.error(f"Round {rnd} failed: {exc}")

    logger.info(f"Finished. {total_saved} new files saved across {len(rounds)} rounds.")


if __name__ == "__main__":
    main()
