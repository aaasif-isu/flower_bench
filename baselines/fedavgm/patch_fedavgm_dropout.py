from pathlib import Path
import re

ROOT = Path(".")
client_path = ROOT / "fedavgm" / "client.py"
server_path = ROOT / "fedavgm" / "server.py"
main_path = ROOT / "fedavgm" / "main.py"

# ----------------------------
# Patch fedavgm/client.py
# ----------------------------
client = client_path.read_text()

if "import random" not in client:
    client = client.replace("import math\n", "import math\nimport random\n")

helper = '''


def should_drop_client(config: dict) -> bool:
    """Randomly simulate client dropout for the current round."""
    dropout_ratio = float(
        config.get("client_dropout_ratio", config.get("client-dropout-ratio", 0.0))
    )

    if dropout_ratio <= 0.0:
        return False

    return random.random() < dropout_ratio
'''

if "def should_drop_client" not in client:
    client = client.replace(
        "from keras.utils import to_categorical\n",
        "from keras.utils import to_categorical\n" + helper,
    )

old_fit_start = '''    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        self.model.set_weights(parameters)
'''

new_fit_start = '''    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""

        if should_drop_client(config):
            server_round = config.get("server_round", config.get("round", "unknown"))
            print(
                f"[CLIENT DROPOUT] FedAvgM client dropped in round {server_round}",
                flush=True,
            )
            raise RuntimeError("Simulated random client dropout")

        self.model.set_weights(parameters)
'''

if "[CLIENT DROPOUT] FedAvgM client dropped" not in client:
    if old_fit_start not in client:
        raise RuntimeError("Could not find expected fit() start block in client.py")
    client = client.replace(old_fit_start, new_fit_start)

client_path.write_text(client)


# ----------------------------
# Patch fedavgm/server.py
# ----------------------------
server = server_path.read_text()

old_server_block = '''    def fit_config_fn(server_round: int):  # pylint: disable=unused-argument
        return {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
        }
'''

new_server_block = '''    def fit_config_fn(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "client_dropout_ratio": float(
                config.get(
                    "client_dropout_ratio",
                    config.get("client-dropout-ratio", 0.1),
                )
            ),
        }
'''

if '"client_dropout_ratio"' not in server:
    if old_server_block not in server:
        raise RuntimeError("Could not find expected fit_config_fn block in server.py")
    server = server.replace(old_server_block, new_server_block)

server_path.write_text(server)


# ----------------------------
# Patch fedavgm/main.py
# ----------------------------
main = main_path.read_text()

if "accept_failures=True" not in main:
    target = '''        min_available_clients=cfg.num_clients,
        evaluate_fn=evaluate_fn,
'''
    replacement = '''        min_available_clients=cfg.num_clients,
        accept_failures=True,
        evaluate_fn=evaluate_fn,
'''
    if target not in main:
        raise RuntimeError("Could not find strategy kwargs block in main.py")
    main = main.replace(target, replacement)

main_path.write_text(main)


# ----------------------------
# Patch Hydra config
# ----------------------------
conf_candidates = [
    ROOT / "fedavgm" / "conf" / "base.yaml",
    ROOT / "conf" / "base.yaml",
    ROOT / "fedavgm" / "conf" / "base.yml",
    ROOT / "conf" / "base.yml",
]

conf_path = next((p for p in conf_candidates if p.exists()), None)

if conf_path is None:
    print("WARNING: Could not find base.yaml/base.yml config file.")
    print("Manually add this under the client section:")
    print("  client_dropout_ratio: 0.1")
else:
    conf = conf_path.read_text()

    if "client_dropout_ratio:" not in conf:
        lines = conf.splitlines()
        out = []
        inserted = False
        in_client = False

        for i, line in enumerate(lines):
            out.append(line)

            if re.match(r"^client:\s*$", line):
                in_client = True
                continue

            if in_client:
                # Insert before next top-level section
                next_is_top_level = (
                    i + 1 < len(lines)
                    and re.match(r"^[A-Za-z0-9_\\-]+:\s*", lines[i + 1])
                    and not lines[i + 1].startswith(" ")
                )
                if next_is_top_level:
                    out.append("  client_dropout_ratio: 0.1")
                    inserted = True
                    in_client = False

        if in_client and not inserted:
            out.append("  client_dropout_ratio: 0.1")
            inserted = True

        if not inserted:
            out.append("")
            out.append("client:")
            out.append("  client_dropout_ratio: 0.1")

        conf_path.write_text("\\n".join(out) + "\\n")
        print(f"Patched config: {conf_path}")
    else:
        print(f"Config already has client_dropout_ratio: {conf_path}")


print("FedAvgM dropout patch complete.")
