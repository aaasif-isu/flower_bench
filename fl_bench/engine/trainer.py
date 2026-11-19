"""Shared client training, model evaluation, and synchronous FL coordinator."""

import csv
import gc
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fl_bench.config import ExperimentConfig
from fl_bench.datasets import (
    FederatedData,
    ensure_config_assets,
    inspect_config_assets,
    load_federated_dataset,
    load_schedule,
)
from fl_bench.methods import (
    BaseMethod,
    ClientResult,
    StateDict,
    create_method,
)
from fl_bench.metrics.schema import (
    build_csv_fields,
    compute_round_comm_bytes,
)
from fl_bench.models import create_model, state_dict_num_bytes, trainable_num_bytes
from fl_bench.utils.io import (
    RunLock,
    TeeLogger,
    verify_metrics_csv,
    write_json_atomic,
    write_yaml_atomic,
)
from fl_bench.utils.provenance import build_run_metadata
from fl_bench.utils.seed import client_round_seed, set_seed


def train_client(
    cid: int,
    global_state: StateDict,
    client_dataset: Dataset,
    device: torch.device,
    config: ExperimentConfig,
    seed: int,
    mu: float = 0.0,
    c_global: Optional[StateDict] = None,
    c_local: Optional[StateDict] = None,
) -> ClientResult:
    """
    Execute local SGD training for a single client.

    Supports:
    - Standard SGD with momentum and weight decay (FedAvg, FedAvgM, FedAdam, FedNova)
    - Proximal penalty `(mu / 2) * ||w - w_global||^2` (FedProx)
    - Per-step control variate gradient correction `g <- g + c_global - c_local` (SCAFFOLD)
    - Optional gradient norm clipping (`config.max_grad_norm`, used for Shakespeare LSTM)
    """
    set_seed(seed)

    assert config.model is not None
    assert config.batch_size is not None
    assert config.local_epochs is not None
    assert config.lr is not None
    assert config.momentum is not None
    assert config.weight_decay is not None

    model = create_model(config.model).to(device)
    model.load_state_dict(global_state)

    global_trainable: Optional[StateDict] = None
    if mu > 0.0:
        trainable_names = dict(model.named_parameters())
        global_trainable = {
            name: value.detach().to(device)
            for name, value in global_state.items()
            if name in trainable_names
        }

    cg_device: Optional[StateDict] = None
    cl_device: Optional[StateDict] = None
    if c_global is not None and c_local is not None:
        cg_device = {name: value.to(device) for name, value in c_global.items()}
        cl_device = {name: value.to(device) for name, value in c_local.items()}

    loader = DataLoader(
        client_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    local_counter = 0.0
    local_normalizing_vec = 0.0
    local_steps = 0

    target_steps = int(config.local_steps) if config.local_steps is not None else None
    max_passes = (
        max(1, math.ceil(target_steps / max(1, len(loader))))
        if target_steps is not None
        else int(config.local_epochs)
    )

    for _ in range(max_passes):
        for x, y in loader:
            if target_steps is not None and local_steps >= target_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)
            task_loss = criterion(logits, y)

            if global_trainable is not None:
                proximal_term = torch.zeros((), device=device)
                for name, param in model.named_parameters():
                    proximal_term = proximal_term + torch.sum(
                        (param - global_trainable[name]) ** 2
                    )
                loss = task_loss + 0.5 * mu * proximal_term
            else:
                loss = task_loss

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                raise FloatingPointError(
                    f"Client {cid} encountered nonfinite training loss ({loss_val}) "
                    f"at local step {local_steps + 1} (lr={config.lr}, momentum={config.momentum})."
                )

            loss.backward()

            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.max_grad_norm,
                )

            if cg_device is not None and cl_device is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad.add_(cg_device[name] - cl_device[name])

            optimizer.step()

            if config.momentum != 0:
                local_counter = local_counter * config.momentum + 1.0
                local_normalizing_vec += local_counter
            else:
                local_normalizing_vec += 1.0

            local_steps += 1

            batch_n = y.size(0)
            total_loss += loss_val * batch_n
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_seen += batch_n

        if target_steps is not None and local_steps >= target_steps:
            break

    local_state: StateDict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    c_delta: Optional[StateDict] = None
    if c_global is not None:
        if local_steps == 0:
            raise RuntimeError(f"Client {cid} performed zero local steps")
        coef = 1.0 / (local_steps * config.lr)
        c_delta = {}
        with torch.no_grad():
            for name in c_global:
                delta64 = (
                    local_state[name].to(torch.float64)
                    - global_state[name].to(torch.float64)
                )
                c_del64 = -c_global[name].to(torch.float64) - coef * delta64
                if not torch.isfinite(c_del64).all():
                    raise FloatingPointError(
                        f"Client {cid} produced nonfinite SCAFFOLD control-variate update "
                        f"for parameter '{name}' (local_steps={local_steps}, lr={config.lr})."
                    )
                c_delta[name] = c_del64.to(c_global[name].dtype)

    return ClientResult(
        state=local_state,
        num_examples=total_seen,
        train_loss=total_loss / total_seen,
        train_accuracy=total_correct / total_seen,
        local_normalizer=local_normalizing_vec,
        local_steps=local_steps,
        c_delta=c_delta,
        cid=cid,
        forward_passes=local_steps,
        backward_passes=local_steps,
    )



@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[float, float, float]:
    """Evaluate `model` on `dataset`, returning `(loss, accuracy, perplexity)`."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch_n = y.size(0)
        total_loss += loss.item() * batch_n
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += batch_n

    mean_loss = total_loss / total_seen
    mean_acc = total_correct / total_seen
    perplexity = math.exp(min(mean_loss, 50.0))

    return mean_loss, mean_acc, perplexity


def run_experiment(config: ExperimentConfig) -> Path:
    """
    Run a complete synchronous Federated Learning experiment and record:
    - `metrics.csv`
    - `config.yaml`
    - `metadata.json`
    - `run.log`
    inside the structured run directory `config.resolve_run_dir()`.
    """
    config.resolve_defaults()
    assert config.model is not None
    assert config.rounds is not None
    assert config.eval_interval is not None

    run_dir = config.resolve_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    partial_csv = run_dir / "metrics.partial.csv"
    lock_file = run_dir / "metrics.lock"
    config_yaml = run_dir / "config.yaml"
    metadata_json = run_dir / "metadata.json"
    run_log = run_dir / "run.log"

    explicit_csv: Optional[Path] = None
    if config.output and config.output.endswith(".csv"):
        candidate = Path(config.output)
        if candidate.resolve() != metrics_csv.resolve():
            explicit_csv = candidate

    with RunLock(lock_file):
        if not config.overwrite:
            done, finite, _ = verify_metrics_csv(
                metrics_csv, config.rounds, config.eval_interval
            )
            if done and finite:
                print(
                    f"Skipping completed run (use --overwrite to re-run): {metrics_csv}",
                    flush=True,
                )
                return metrics_csv
            if explicit_csv is not None and explicit_csv.exists():
                done_exp, finite_exp, _ = verify_metrics_csv(
                    explicit_csv, config.rounds, config.eval_interval
                )
                if done_exp and finite_exp:
                    print(
                        f"Skipping completed run (use --overwrite to re-run): {explicit_csv}",
                        flush=True,
                    )
                    return explicit_csv

        write_yaml_atomic(config_yaml, config.to_dict())
        start_wall = time.time()

        with TeeLogger(run_log) as logger:
            set_seed(config.training_seed)

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            gpu_name = (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            )
            logger.log(f"Device: {device}")
            if gpu_name:
                logger.log(f"GPU: {gpu_name}")

            ensure_config_assets(config, log_fn=logger.log)

            data = load_federated_dataset(config)
            config.num_clients = data.num_clients

            schedule = load_schedule(
                config.schedule_file,
                num_clients=config.num_clients,
                clients_per_round=config.clients_per_round,
                num_rounds=config.rounds,
                seed=config.schedule_seed,
            )

            method = create_method(config)
            global_model = method.create_global_model(device)
            trainable_keys = set(dict(global_model.named_parameters()).keys())
            model_bytes = state_dict_num_bytes(global_model)
            trainable_bytes = trainable_num_bytes(global_model)

            method.init_server(global_model)

            fields = build_csv_fields(
                data,
                method,
                include_comm_accounting=bool(config.include_comm_accounting),
            )

            cum_fit_clients = 0
            cum_up_bytes = 0
            cum_down_bytes = 0
            cum_local_steps = 0
            cum_fwd_passes = 0
            cum_bwd_passes = 0

            with partial_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

                test_loss, test_acc, test_ppl = evaluate(
                    global_model,
                    data.test_dataset,
                    device,
                    batch_size=config.eval_batch_size,
                )

                if data.include_perplexity:
                    logger.log(
                        f"Round 0 | test_loss={test_loss:.4f} "
                        f"test_acc={test_acc:.4f} ppl={test_ppl:.2f}"
                    )
                else:
                    logger.log(
                        f"Round 0 | test_loss={test_loss:.4f} "
                        f"test_acc={test_acc:.4f}"
                    )

                for round_id in range(1, config.rounds + 1):
                    method.current_round_id = round_id
                    selected = schedule[str(round_id)]
                    logger.log(f"Round {round_id} | selected={selected}")

                    global_state: StateDict = {
                        key: value.detach().cpu().clone()
                        for key, value in global_model.state_dict().items()
                    }

                    client_results: List[ClientResult] = []
                    for cid in selected:
                        seed = client_round_seed(
                            config.training_seed, round_id, cid
                        )
                        result = method.run_client(
                            cid=cid,
                            global_state=global_state,
                            client_dataset=data.train_clients[cid],
                            device=device,
                            seed=seed,
                        )
                        client_results.append(result)

                    new_state, extra_metrics = method.aggregate(
                        round_id=round_id,
                        global_state=global_state,
                        client_results=client_results,
                        trainable_keys=trainable_keys,
                    )
                    global_model.load_state_dict(new_state)

                    total_examples = sum(r.num_examples for r in client_results)
                    round_steps = sum(r.local_steps for r in client_results)
                    round_fwd = sum(
                        (r.forward_passes if r.forward_passes > 0 else r.local_steps)
                        for r in client_results
                    )
                    round_bwd = sum(
                        (r.backward_passes if r.backward_passes > 0 else r.local_steps)
                        for r in client_results
                    )
                    round_extra_up = sum(r.extra_upload_bytes for r in client_results)
                    round_extra_down = sum(r.extra_download_bytes for r in client_results)
                    has_dynamic_comm = any(
                        (r.extra_upload_bytes > 0 or r.extra_download_bytes > 0)
                        for r in client_results
                    )
                    train_loss = (
                        sum(
                            r.train_loss * r.num_examples
                            for r in client_results
                        )
                        / total_examples
                    )
                    train_acc = (
                        sum(
                            r.train_accuracy * r.num_examples
                            for r in client_results
                        )
                        / total_examples
                    )

                    if (
                        round_id % config.eval_interval == 0
                        or round_id == config.rounds
                    ):
                        test_loss, test_acc, test_ppl = evaluate(
                            global_model,
                            data.test_dataset,
                            device,
                            batch_size=config.eval_batch_size,
                        )
                        if not (math.isfinite(test_loss) and math.isfinite(test_acc)):
                            raise FloatingPointError(
                                f"Round {round_id}: Nonfinite evaluation metrics "
                                f"(test_loss={test_loss}, test_acc={test_acc})."
                            )
                    else:
                        test_loss = float("nan")
                        test_acc = float("nan")
                        test_ppl = float("nan")

                    if not math.isfinite(train_loss):
                        raise FloatingPointError(
                            f"Round {round_id}: Nonfinite aggregate train_loss ({train_loss})."
                        )

                    log_parts = [
                        f"Round {round_id} |",
                        f"train_loss={train_loss:.4f}",
                        f"train_acc={train_acc:.4f}",
                        f"test_loss={test_loss:.4f}",
                        f"test_acc={test_acc:.4f}",
                    ]
                    if data.include_perplexity:
                        log_parts.append(f"ppl={test_ppl:.2f}")

                    extra_log = method.format_extra_log(extra_metrics)
                    if extra_log:
                        log_parts.append(extra_log)

                    if data.include_num_fit_examples:
                        log_parts.append(f"examples={total_examples}")

                    logger.log(" ".join(log_parts))

                    row: Dict[str, object] = {
                        "round": round_id,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "num_fit_clients": len(selected),
                        "run_completed": (round_id == config.rounds),
                    }
                    if data.include_perplexity:
                        row["test_perplexity"] = test_ppl
                    if (
                        data.include_num_fit_examples
                        or config.include_comm_accounting
                    ):
                        row["num_fit_examples"] = total_examples
                    row.update(extra_metrics)

                    if config.include_comm_accounting:
                        if any(
                            (r.upload_bytes_override is not None or r.download_bytes_override is not None)
                            for r in client_results
                        ):
                            up_b = sum(
                                int(r.upload_bytes_override if r.upload_bytes_override is not None else model_bytes)
                                for r in client_results
                            )
                            down_b = sum(
                                int(r.download_bytes_override if r.download_bytes_override is not None else model_bytes)
                                for r in client_results
                            )
                        else:
                            up_b, down_b = compute_round_comm_bytes(
                                len(selected),
                                model_bytes,
                                method=config.method,
                                trainable_bytes=trainable_bytes,
                                round_id=round_id,
                                extra_upload_bytes=round_extra_up if has_dynamic_comm else None,
                                extra_download_bytes=round_extra_down if has_dynamic_comm else None,
                                fedlesam_variant=str(getattr(config, "fedlesam_variant", "base") or "base"),
                                model_name=str(config.model or "resnet18"),
                            )
                        cum_fit_clients += len(selected)
                        cum_up_bytes += up_b
                        cum_down_bytes += down_b
                        cum_local_steps += round_steps
                        cum_fwd_passes += round_fwd
                        cum_bwd_passes += round_bwd
                        row["cumulative_fit_clients"] = cum_fit_clients
                        row["round_upload_bytes"] = up_b
                        row["round_download_bytes"] = down_b
                        row["round_local_steps"] = round_steps
                        row["round_forward_passes"] = round_fwd
                        row["round_backward_passes"] = round_bwd
                        row["cumulative_upload_bytes"] = cum_up_bytes
                        row["cumulative_download_bytes"] = cum_down_bytes
                        row["cumulative_local_steps"] = cum_local_steps
                        row["cumulative_forward_passes"] = cum_fwd_passes
                        row["cumulative_backward_passes"] = cum_bwd_passes

                    writer.writerow(row)
                    f.flush()

                    if method.cleanup_after_round:
                        del client_results
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            elapsed = time.time() - start_wall
            partial_csv.replace(metrics_csv)

            if explicit_csv is not None:
                explicit_csv.parent.mkdir(parents=True, exist_ok=True)
                explicit_csv.write_bytes(metrics_csv.read_bytes())

            rows_done, finite_ok, n_rows = verify_metrics_csv(
                metrics_csv, config.rounds, config.eval_interval
            )
            meta = build_run_metadata(
                config.to_dict(),
                run_completed=rows_done,
                metrics_finite=finite_ok,
                completed_rounds=n_rows,
                elapsed_seconds=elapsed,
                device_str=str(device),
                gpu_name=gpu_name,
            )
            write_json_atomic(metadata_json, meta)
            logger.log(f"Saved: {metrics_csv}")

    return explicit_csv if explicit_csv is not None else metrics_csv


def execute_grid(configs: Sequence[ExperimentConfig]) -> List[Path]:
    """
    Validate existing assets across the grid, then execute each resolved
    `ExperimentConfig` run in order (generating any missing assets once and
    reusing them across methods sharing the same condition and seed).
    """
    for cfg in configs:
        part_status, sched_status = inspect_config_assets(cfg)
        if part_status.action == "INVALID":
            raise ValueError(
                part_status.error or f"Invalid partition asset: {part_status.path}"
            )
        if sched_status.action == "INVALID":
            raise ValueError(
                sched_status.error or f"Invalid schedule asset: {sched_status.path}"
            )

    outputs: List[Path] = []
    for cfg in configs:
        outputs.append(run_experiment(cfg))
    return outputs
