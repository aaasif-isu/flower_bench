"""
Client capacity profile parsing and fixed/dynamic width-rate scheduling for HeteroFL
(Diao et al., ICLR 2021).
Reference: https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/blob/master/src/utils.py
and `src/fed.py`.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

MODEL_SPLIT_RATE: Dict[str, float] = {
    "a": 1.0,
    "b": 0.5,
    "c": 0.25,
    "d": 0.125,
    "e": 0.0625,
}


def parse_model_mode(
    model_mode: str,
    custom_split_rates: Dict[str, float] | None = None,
) -> Tuple[float, List[float], List[int], List[float]]:
    """
    Parse a HeteroFL `model_mode` string (e.g., `'a1'`, `'a2-b8'`, `'a1-b1-c1-d1-e1'`,
    or explicit numeric rates `'1.0:2-0.5:3-0.25:5'`) into:
        `(global_model_rate, mode_rates, raw_proportions, normalized_proportions)`
    matching `process_control()` in `src/utils.py:114-142`.
    """
    rate_map = dict(MODEL_SPLIT_RATE)
    if custom_split_rates:
        for k, v in custom_split_rates.items():
            rate_map[str(k).strip().lower()] = float(v)

    tokens = [t.strip() for t in str(model_mode).strip().split("-") if t.strip()]
    if not tokens:
        raise ValueError(f"Invalid empty HeteroFL model_mode: '{model_mode}'")

    mode_rates: List[float] = []
    proportions: List[int] = []

    for tok in tokens:
        if ":" in tok:
            r_str, p_str = tok.split(":", 1)
            r_val = rate_map.get(r_str.strip().lower(), float(r_str.strip()))
            p_val = int(p_str.strip())
        else:
            tier_char = tok[0].lower()
            if tier_char not in rate_map:
                raise ValueError(
                    f"Unknown HeteroFL capacity tier '{tier_char}' in token '{tok}'. "
                    f"Supported tiers: {sorted(rate_map)} or numeric '<rate>:<weight>'."
                )
            r_val = float(rate_map[tier_char])
            p_val = int(tok[1:]) if len(tok) > 1 else 1

        if not (0.0 < r_val <= 1.0):
            raise ValueError(f"HeteroFL model rate must be in (0, 1], got {r_val} in '{tok}'.")
        if p_val <= 0:
            raise ValueError(f"HeteroFL tier proportion weight must be positive, got {p_val} in '{tok}'.")

        mode_rates.append(r_val)
        proportions.append(p_val)

    global_model_rate = float(mode_rates[0])
    total_p = float(sum(proportions))
    norm_proportions = [p / total_p for p in proportions]
    return global_model_rate, mode_rates, proportions, norm_proportions


def assign_client_model_rates(
    num_clients: int,
    split_mode: str,
    model_mode: str,
    *,
    seed: int = 42,
    round_id: int = 1,
    custom_split_rates: Dict[str, float] | None = None,
) -> Tuple[float, List[float]]:
    """
    Assign a subnetwork width rate `r_i` to each client `i in 0..num_clients-1`
    matching `process_control()` (`src/utils.py:124-141`) and `Federation.make_model_rate()`
    (`src/fed.py:15-24`).

    - `split_mode == "fix"`: Deterministic block assignment across clients `0..N-1`
      (`num_users_proportion = N // sum(proportion)`).
    - `split_mode == "dynamic"`: Per-round multinomial sampling over `norm_proportions`.
    """
    global_rate, mode_rates, proportions, norm_props = parse_model_mode(
        model_mode, custom_split_rates=custom_split_rates
    )
    mode = str(split_mode).strip().lower()

    if mode == "fix":
        unit = num_clients // sum(proportions)
        rates: List[float] = []
        for r_val, p_val in zip(mode_rates, proportions):
            rates.extend([float(r_val)] * (unit * p_val))
        while len(rates) < num_clients:
            rates.append(float(mode_rates[-1]))
        return global_rate, rates[:num_clients]

    if mode == "dynamic":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(round_id) * 1009)
        probs = torch.tensor(norm_props, dtype=torch.float64)
        rate_idx = torch.multinomial(
            probs, num_samples=int(num_clients), replacement=True, generator=gen
        ).tolist()
        return global_rate, [float(mode_rates[idx]) for idx in rate_idx]

    raise ValueError(
        f"Invalid HeteroFL `model_split_mode` '{split_mode}'. Expected 'fix' or 'dynamic'."
    )
