import os
import csv
import time
import numpy as np
from hydra import initialize, compose
from hydra.utils import instantiate
import fedmeta.main as main_mod
from fedmeta.client import gen_client_fn
from fedmeta.strategy import fedmeta_update_maml
from flwr.server.strategy.aggregate import aggregate

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FEDMETA_MAX_USERS"] = "20"

DATA_PATH = "/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/external/leaf/data/femnist/data"
CSV_PATH = "fedmeta_leaf_femnist_niid_20c_10cpr_50r_direct.csv"

NUM_ROUNDS = 50
CLIENTS_PER_ROUND = 10
BETA = 0.0001

def mb(arrs):
    return sum(a.nbytes for a in arrs) / (1024 * 1024)

def unflatten_like(flat, like_params):
    """Rebuild flattened grads into the same shapes as model params."""
    flat = np.asarray(flat, dtype=np.float32).ravel()
    rebuilt = []
    idx = 0
    for arr in like_params:
        size = arr.size
        rebuilt.append(flat[idx:idx + size].reshape(arr.shape))
        idx += size
    return rebuilt

with initialize(version_base=None, config_path="fedmeta/conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            f"path={DATA_PATH}",
            "algo=fedmeta_maml",
            "data=femnist",
            "data.gradient_step=1",
            "data.batch_size=10",
        ],
    )

trainloaders, valloaders, _ = main_mod.load_datasets(config=cfg.data, path=cfg.path)
print("Loaded clients:", len(trainloaders["sup"]), flush=True)

client_fn = gen_client_fn(
    num_epochs=cfg.num_epochs,
    trainloaders=trainloaders,
    valloaders=valloaders,
    learning_rate=cfg.algo.femnist.beta,
    model=cfg.data.model,
    gradient_step=cfg.data.gradient_step,
)

c0 = client_fn("0").numpy_client
global_params = c0.get_parameters({})

server_net = instantiate(cfg.data.model)
cumulative_comm = 0.0
cumulative_time = 0.0

fields = [
    "method","dataset","model","round",
    "train_loss","test_loss","test_accuracy","num_fit_clients","fit_failures",
    "fit_param_down_mb","fit_param_up_mb","fit_param_total_comm_mb",
    "round_total_comm_mb","cumulative_total_comm_mb",
    "round_wall_time_sec","cumulative_round_wall_time_sec"
]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for rnd in range(1, NUM_ROUNDS + 1):
        start = time.perf_counter()
        selected = [((rnd - 1) * CLIENTS_PER_ROUND + i) % len(trainloaders["sup"]) for i in range(CLIENTS_PER_ROUND)]

        results = []
        losses = []
        failures = 0

        down_mb = mb(global_params) * len(selected)

        for cid in selected:
            print(f'Round {rnd}: starting client {cid}', flush=True)
            try:
                client = client_fn(str(cid)).numpy_client
                params_prime, n, metrics = client.fit(
                    global_params,
                    {
                        "algo": "fedmeta_maml",
                        "alpha": 0.001,
                        "beta": BETA,
                        "data": "femnist",
                    },
                )
                if "grads" in metrics:
                    metrics["grads"] = unflatten_like(metrics["grads"], global_params)
                results.append((params_prime, int(n), metrics))
                print(f'Round {rnd}: finished client {cid}', flush=True)
                losses.append(float(metrics["loss"]))
            except Exception as e:
                failures += 1
                print("FAILED client", cid, repr(e), flush=True)

        if not results:
            print("All clients failed on round", rnd, flush=True)
            break

        weights_results = [(p, n) for p, n, m in results]
        grads_results = [(m["grads"], n) for p, n, m in results]

        print(f'Round {rnd}: aggregating gradients', flush=True)
        gradients_agg = aggregate(grads_results)
        print(f'Round {rnd}: gradient aggregation OK', flush=True)

        print(f'Round {rnd}: updating server model', flush=True)
        global_params = fedmeta_update_maml(
            server_net,
            BETA,
            weights_results[0][0],
            gradients_agg,
            0.001,
        )

        print(f'Round {rnd}: evaluating global model', flush=True)
        eval_losses = []
        eval_accs = []
        for eval_cid in range(len(valloaders["sup"])):
            eval_client = client_fn(str(eval_cid)).numpy_client
            eval_loss, eval_n, eval_metrics = eval_client.evaluate(
                global_params,
                {
                    "algo": "fedmeta_maml",
                    "alpha": 0.001,
                    "beta": BETA,
                    "data": "femnist",
                },
            )
            eval_losses.append(float(eval_loss))
            eval_accs.append(float(eval_metrics["correct"]))
        test_loss = float(np.mean(eval_losses))
        test_accuracy = float(np.mean(eval_accs))
        print(f'Round {rnd}: eval accuracy={test_accuracy:.4f}', flush=True)

        up_mb = 0.0
        for p, n, m in results:
            up_mb += mb(p)
            up_mb += mb(m["grads"])

        round_comm = down_mb + up_mb
        cumulative_comm += round_comm

        round_time = time.perf_counter() - start
        cumulative_time += round_time

        print(f'Round {rnd}: writing CSV row', flush=True)
        row = {
            "method": "FedMeta-Direct",
            "dataset": "LEAF-FEMNIST-NIID",
            "model": "FemnistNetwork",
            "round": rnd,
            "train_loss": float(np.mean(losses)),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "num_fit_clients": len(results),
            "fit_failures": failures,
            "fit_param_down_mb": down_mb,
            "fit_param_up_mb": up_mb,
            "fit_param_total_comm_mb": round_comm,
            "round_total_comm_mb": round_comm,
            "cumulative_total_comm_mb": cumulative_comm,
            "round_wall_time_sec": round_time,
            "cumulative_round_wall_time_sec": cumulative_time,
        }

        writer.writerow(row)
        f.flush()

        print(f"Round {rnd}/{NUM_ROUNDS} loss={row['train_loss']:.4f} clients={len(results)} failures={failures}", flush=True)

print("DONE:", CSV_PATH, flush=True)
