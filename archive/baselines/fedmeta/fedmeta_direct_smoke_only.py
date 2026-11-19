import os
import traceback
from hydra import initialize, compose
import fedmeta.main as main_mod
from fedmeta.client import gen_client_fn

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FEDMETA_MAX_USERS"] = "20"

try:
    with initialize(version_base=None, config_path="fedmeta/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "path=/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/external/leaf/data/femnist/data",
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

    c = client_fn("0").numpy_client
    params = c.get_parameters({})

    print("Starting client fit", flush=True)
    out = c.fit(params, {
        "algo": "fedmeta_maml",
        "alpha": 0.001,
        "beta": 0.0001,
        "data": "femnist",
    })

    print("FIT OK", flush=True)
    print("num_examples:", out[1], flush=True)
    print("metrics keys:", list(out[2].keys()), flush=True)
    print("loss:", out[2].get("loss"), flush=True)

except Exception:
    print("FAILED", flush=True)
    traceback.print_exc()
