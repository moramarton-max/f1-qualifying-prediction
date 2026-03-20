from typing import Dict


ERA_DEFINITIONS: Dict[str, list] = {
    "pre_2022": list(range(2018, 2022)),
    "hybrid_v2": list(range(2022, 2026)),
    "new_regs": [2026],
}

ERA_WEIGHTS: Dict[str, float] = {
    "pre_2022": 0.3,
    "hybrid_v2": 1.0,
    "new_regs": 1.0,
}


def get_era(year: int) -> str:
    for era, years in ERA_DEFINITIONS.items():
        if year in years:
            return era
    return "hybrid_v2"  # default fallback


def get_sample_weight(year: int) -> float:
    return ERA_WEIGHTS.get(get_era(year), 1.0)
