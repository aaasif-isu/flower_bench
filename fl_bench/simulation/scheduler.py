"""
Deterministic client cohort and concurrency scheduler for Track 3 simulation.

Integrates existing round schedules (`fl_bench/datasets/schedule.py`) with
availability filtering, active in-flight concurrency tracking, and reproducible
replacement sampling for synchronous, asynchronous (`fedasync`), and buffered
asynchronous (`fedbuff`) execution strategies.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from fl_bench.simulation.profiles import ClientProfileManager


class ClientScheduler:
    """
    Deterministic client selection and concurrency manager.

    - Reuses canonical round schedules for synchronous rounds (and initial async
      dispatch at t=0) when scheduled clients are available.
    - Samples replacement clients deterministically without replacement when
      scheduled clients are unavailable or when dispatching continuous jobs in
      `fedasync` / `fedbuff`.
    - Ensures a single client never runs two overlapping jobs concurrently.
    """

    def __init__(
        self,
        *,
        num_clients: int,
        max_concurrency: int,
        profile_manager: ClientProfileManager,
        schedule_seed: int = 42,
        base_schedule: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}.")
        if max_concurrency <= 0 or max_concurrency > num_clients:
            raise ValueError(
                f"max_concurrency ({max_concurrency}) must be in 1..{num_clients}."
            )
        self.num_clients = int(num_clients)
        self.max_concurrency = int(max_concurrency)
        self.profile_manager = profile_manager
        self.schedule_seed = int(schedule_seed)
        self.base_schedule: Dict[str, List[int]] = dict(base_schedule or {})

        self.in_flight_clients: Set[int] = set()
        self.client_availability_checks: Dict[int, int] = {
            cid: 0 for cid in range(self.num_clients)
        }
        self.client_job_dispatches: Dict[int, int] = {
            cid: 0 for cid in range(self.num_clients)
        }
        self.unavailable_selections: int = 0
        self._dispatch_epoch_counter: int = 0

    def _check_available(self, cid: int) -> bool:
        self.client_availability_checks[cid] += 1
        chk_idx = self.client_availability_checks[cid]
        ok = self.profile_manager.is_client_available(cid, chk_idx)
        if not ok:
            self.unavailable_selections += 1
        return ok

    def mark_dispatched(self, cid: int) -> int:
        """Mark `cid` as in-flight and return its 1-based per-client job index."""
        cid_int = int(cid)
        if cid_int in self.in_flight_clients:
            raise RuntimeError(
                f"Client {cid_int} is already in-flight; cannot dispatch overlapping job."
            )
        self.in_flight_clients.add(cid_int)
        self.client_job_dispatches[cid_int] += 1
        return self.client_job_dispatches[cid_int]

    def mark_completed(self, cid: int) -> None:
        """Release `cid` from the active in-flight set."""
        self.in_flight_clients.discard(int(cid))

    def select_synchronous_cohort(
        self,
        round_id: int,
        cohort_size: Optional[int] = None,
    ) -> Tuple[List[int], int]:
        """
        Select up to `cohort_size` available clients for synchronous `round_id`.
        Returns `(selected_cids, unavailable_count_in_round)`.
        """
        target = int(cohort_size if cohort_size is not None else self.max_concurrency)
        unavail_before = self.unavailable_selections
        selected: List[int] = []
        tried: Set[int] = set()

        scheduled_cids = self.base_schedule.get(str(round_id), [])
        for raw_cid in scheduled_cids:
            if len(selected) >= target:
                break
            cid = int(raw_cid)
            if cid in self.in_flight_clients or cid in tried:
                continue
            tried.add(cid)
            if self._check_available(cid):
                selected.append(cid)

        if len(selected) < target:
            remaining = [
                cid
                for cid in range(self.num_clients)
                if cid not in self.in_flight_clients and cid not in tried
            ]
            if remaining:
                rng = np.random.default_rng([self.schedule_seed, 303, int(round_id)])
                perm = rng.permutation(remaining).tolist()
                for cid in perm:
                    if len(selected) >= target:
                        break
                    tried.add(int(cid))
                    if self._check_available(int(cid)):
                        selected.append(int(cid))

        unavail_delta = self.unavailable_selections - unavail_before
        return selected, unavail_delta

    def select_async_replacements(
        self,
        num_slots: Optional[int] = None,
        use_round1_schedule_if_initial: bool = True,
    ) -> Tuple[List[int], int]:
        """
        Select available non-in-flight clients to fill open concurrency slots in
        `fedasync` or `fedbuff`.
        Returns `(selected_cids, unavailable_count_in_call)`.
        """
        open_slots = (
            int(num_slots)
            if num_slots is not None
            else max(0, self.max_concurrency - len(self.in_flight_clients))
        )
        if open_slots <= 0:
            return [], 0

        unavail_before = self.unavailable_selections
        selected: List[int] = []
        tried: Set[int] = set()

        if (
            use_round1_schedule_if_initial
            and self._dispatch_epoch_counter == 0
            and "1" in self.base_schedule
        ):
            for raw_cid in self.base_schedule["1"]:
                if len(selected) >= open_slots:
                    break
                cid = int(raw_cid)
                if cid in self.in_flight_clients or cid in tried:
                    continue
                tried.add(cid)
                if self._check_available(cid):
                    selected.append(cid)

        while len(selected) < open_slots:
            self._dispatch_epoch_counter += 1
            eligible = [
                cid
                for cid in range(self.num_clients)
                if cid not in self.in_flight_clients
                and cid not in selected
                and cid not in tried
            ]
            if not eligible:
                break
            rng = np.random.default_rng(
                [self.schedule_seed, 404, self._dispatch_epoch_counter]
            )
            perm = rng.permutation(eligible).tolist()
            for cid in perm:
                if len(selected) >= open_slots:
                    break
                tried.add(int(cid))
                if self._check_available(int(cid)):
                    selected.append(int(cid))

        unavail_delta = self.unavailable_selections - unavail_before
        return selected, unavail_delta

