"""Round-by-round client participation schedule generation, validation, and loading."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fl_bench.utils.io import resolve_repo_path


def generate_schedule(
    num_clients: int,
    clients_per_round: int,
    num_rounds: int,
    seed: int,
) -> Dict[str, List[int]]:
    """Deterministically sample participating clients for each round 1..R."""
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}.")
    if clients_per_round <= 0 or clients_per_round > num_clients:
        raise ValueError(
            f"clients_per_round ({clients_per_round}) must be in 1..{num_clients}."
        )
    if num_rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}.")

    rng = np.random.default_rng(seed)
    schedule: Dict[str, List[int]] = {}

    for round_id in range(1, num_rounds + 1):
        selected = rng.choice(
            num_clients,
            size=clients_per_round,
            replace=False,
        )
        schedule[str(round_id)] = [int(x) for x in selected]

    return schedule


def save_schedule(
    output_path: str | Path,
    num_clients: int,
    clients_per_round: int,
    num_rounds: int,
    seed: int,
    overwrite: bool = False,
) -> Dict[str, List[int]]:
    """Atomically generate and write a round schedule JSON file."""
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing schedule file: {output}"
        )

    schedule = generate_schedule(
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        num_rounds=num_rounds,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")

    with tmp.open("w") as f:
        json.dump(
            {
                "num_clients": int(num_clients),
                "clients_per_round": int(clients_per_round),
                "num_rounds": int(num_rounds),
                "seed": int(seed),
                "schedule": schedule,
            },
            f,
            indent=2,
        )
        f.write("\n")

    tmp.replace(output)
    return schedule


def validate_schedule_file(
    path: str | Path,
    num_clients: Optional[int] = None,
    clients_per_round: Optional[int] = None,
    num_rounds: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[int]]:
    """
    Validate a schedule JSON file's metadata and round-by-round client lists.
    Raises `ValueError` if corrupted or incompatible with requested parameters.
    """
    resolved = resolve_repo_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Schedule file not found: {path}")

    try:
        with open(resolved, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupted schedule JSON file '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Corrupted schedule file '{path}': root JSON value must be an object.")

    required_keys = {"num_clients", "clients_per_round", "num_rounds", "seed", "schedule"}
    missing_keys = required_keys - set(payload.keys())
    if missing_keys:
        raise ValueError(
            f"Corrupted schedule file '{path}': missing required keys {sorted(missing_keys)}."
        )

    try:
        file_n = int(payload["num_clients"])
        file_m = int(payload["clients_per_round"])
        file_r = int(payload["num_rounds"])
        file_seed = int(payload["seed"])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Corrupted schedule file '{path}': non-integer metadata fields ({exc})."
        ) from exc

    if num_clients is not None and file_n != int(num_clients):
        raise ValueError(
            f"Incompatible schedule file '{path}': num_clients={file_n}, expected {num_clients}."
        )
    if clients_per_round is not None and file_m != int(clients_per_round):
        raise ValueError(
            f"Incompatible schedule file '{path}': clients_per_round={file_m}, expected {clients_per_round}."
        )
    if num_rounds is not None and file_r != int(num_rounds):
        raise ValueError(
            f"Incompatible schedule file '{path}': num_rounds={file_r}, expected {num_rounds}."
        )
    if seed is not None and file_seed != int(seed):
        raise ValueError(
            f"Incompatible schedule file '{path}': seed={file_seed}, expected {seed}."
        )

    schedule = payload["schedule"]
    if not isinstance(schedule, dict):
        raise ValueError(f"Corrupted schedule file '{path}': 'schedule' must be a JSON object.")

    expected_schedule = generate_schedule(file_n, file_m, file_r, file_seed)
    if schedule != expected_schedule:
        raise ValueError(
            f"Corrupted or modified schedule contents in '{path}': round selections do not "
            f"match deterministic schedule for (N={file_n}, M={file_m}, R={file_r}, seed={file_seed})."
        )

    return schedule


def load_schedule(
    path: str | Path,
    num_clients: Optional[int] = None,
    clients_per_round: Optional[int] = None,
    num_rounds: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Validate and load the `schedule` dictionary from a schedule JSON file."""
    return validate_schedule_file(
        path,
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        num_rounds=num_rounds,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic client participation schedules."
    )
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    schedule = save_schedule(
        output_path=args.output,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        num_rounds=args.rounds,
        seed=args.seed,
    )

    print("Saved:", args.output)
    print("Round 1:", schedule["1"])
    if "2" in schedule:
        print("Round 2:", schedule["2"])
    print(f"Round {args.rounds}:", schedule[str(args.rounds)])


if __name__ == "__main__":
    main()
