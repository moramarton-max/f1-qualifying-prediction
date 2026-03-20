"""Static configuration: weekend formats, session ordering, sprint round lists."""

from typing import Dict, List

# Sessions that precede qualifying, in order, for each weekend type
STANDARD_SESSIONS: List[str] = ["FP1", "FP2", "FP3"]
SPRINT_SESSIONS: List[str] = ["FP1", "SQ"]

# Target sessions
TARGET_QUALI = "Q"
TARGET_SPRINT_QUALI = "SQ"

# Sprint weekend round numbers by year
SPRINT_ROUNDS: Dict[int, List[int]] = {
    2022: [4, 11, 21],
    2023: [4, 10, 17, 18, 20],
    2024: [5, 11, 16, 17, 19],
    2025: [],
}

# FastF1 session identifier strings
SESSION_IDENTIFIERS: Dict[str, str] = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "SQ": "Sprint Qualifying",
    "S": "Sprint",
    "Q": "Qualifying",
}


def is_sprint_weekend(year: int, round_number: int) -> bool:
    return round_number in SPRINT_ROUNDS.get(year, [])


def get_preceding_sessions(year: int, round_number: int, target: str) -> List[str]:
    """Return ordered list of sessions that precede the target session."""
    if target == TARGET_QUALI:
        if is_sprint_weekend(year, round_number):
            return SPRINT_SESSIONS
        return STANDARD_SESSIONS
    if target == TARGET_SPRINT_QUALI:
        return ["FP1"]
    return []
