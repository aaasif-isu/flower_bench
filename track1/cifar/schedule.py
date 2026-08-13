from pathlib import Path
import argparse
import json

import numpy as np


def generate_schedule(
    num_clients,
    clients_per_round,
    num_rounds,
    seed,
):
    rng = np.random.default_rng(seed)

    schedule = {}

    for round_id in range(1, num_rounds + 1):
        selected = rng.choice(
            num_clients,
            size=clients_per_round,
            replace=False,
        )

        schedule[str(round_id)] = [
            int(x) for x in selected
        ]

    return schedule


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    schedule = generate_schedule(
        args.num_clients,
        args.clients_per_round,
        args.rounds,
        args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as f:
        json.dump(
            {
                "num_clients": args.num_clients,
                "clients_per_round": args.clients_per_round,
                "num_rounds": args.rounds,
                "seed": args.seed,
                "schedule": schedule,
            },
            f,
            indent=2,
        )

    print("Saved:", output)
    print("Round 1:", schedule["1"])
    print("Round 2:", schedule["2"])
    print(
        f"Round {args.rounds}:",
        schedule[str(args.rounds)],
    )


if __name__ == "__main__":
    main()
