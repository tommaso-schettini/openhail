from typing import Tuple

import numpy as np

ChargingInfrastructure = Tuple[np.ndarray, np.ndarray]

CHARGERS = 0
REPOS = 1


def infrastructure_string(solution: ChargingInfrastructure):
    """Generate a string representation of the charging infrastructure solution."""
    return f"S: {(solution[CHARGERS].nonzero()[0])} | {(solution[REPOS].nonzero()[0])}"


def get_rng(seed: int):
    return np.random.RandomState(seed)


def parse_infrastructure(config: dict, num_candidates: int) -> ChargingInfrastructure:

    infrastructure_config = config["infrastructure"]
    if infrastructure_config["mode"] == "select":
        seed = infrastructure_config.get("seed", 1234)
        return get_random_infrastructure(num_candidates, infrastructure_config, seed)

    raise ValueError(
        f"Unrecognized infrastructure mode {infrastructure_config['mode']}"
    )


def get_random_infrastructure(
    num_lots: int, charger_config: dict, seed: int
) -> ChargingInfrastructure:
    """Get a random infrastructure given a charger configuration."""
    rng = get_rng(seed)
    repo_locations = charger_config["locations"]
    repo_count = charger_config.get("repo_count", 100)
    charger_locations = charger_config["selection"]
    charger_count = charger_config.get("count", 1)

    repos_idxs, repos_binary = draw_candidates(num_lots, repo_locations, rng=rng)
    _, chargers_binary = draw_candidates(
        num_lots, charger_locations, rng=rng, pool=repos_idxs
    )
    charger_locations = charger_count * chargers_binary
    repos = repo_count * repos_binary
    return charger_locations, repos


def get_infrastructure(
    charger_idxs: np.ndarray,
    repo_idxs: np.ndarray,
    num_candidates: int,
    charger_config: dict,
) -> ChargingInfrastructure:
    """Get an infrastructure given a list of charger and repo indices."""
    repo_count = charger_config.get("repo_count", 100)
    charger_count = charger_config.get("count", 1)
    chargers = np.zeros(num_candidates, dtype=bool)
    repos = np.zeros(num_candidates, dtype=bool)
    chargers[charger_idxs] = 1
    repos[repo_idxs] = 1
    chargers = charger_count * chargers
    repos = repo_count * repos
    return chargers, repos


def draw_candidates(num_candidates, count, rng, pool=None):
    if pool is None:
        pool = np.arange(num_candidates)
    solution = np.zeros(num_candidates, dtype=bool)
    indices = np.sort(rng.choice(pool, size=count, replace=False))
    solution[indices] = 1
    return indices, solution
