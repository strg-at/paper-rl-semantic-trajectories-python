import os
import time
import zlib
import json


def create_experiment_hash(n_steps: int, gamma: float, save_path: str) -> tuple[str, str]:
    exp_variables = {"n_steps": n_steps, "gamma": gamma}
    json_str = json.dumps(exp_variables, sort_keys=True, indent=4)
    hyperparam_hash = zlib.adler32(json_str.encode("utf-8"))
    hyperparam_hash = f"{hyperparam_hash:08x}"

    experiment_path = os.path.join(save_path, f"trajectories_{hyperparam_hash}")
    output_dir = os.path.join(experiment_path, f"run_{round(time.time())}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(experiment_path, "parameters.json")) as f:
        f.write(json_str)

    return output_dir, json_str
