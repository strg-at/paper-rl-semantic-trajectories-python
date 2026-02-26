import json
import os
import time
import zlib


def get_params_hash_str(params: dict) -> str:
    json_str = json.dumps(params, sort_keys=True, indent=4)
    hyperparam_hash = zlib.adler32(json_str.encode("utf-8"))
    hyperparam_hash = f"{hyperparam_hash:08x}"
    return hyperparam_hash


def create_experiment_hash_dir(params: dict, save_path: str) -> str:
    hyperparam_hash = get_params_hash_str(params)

    experiment_path = os.path.join(save_path, f"trajectories_{hyperparam_hash}")
    output_dir = os.path.join(experiment_path, f"run_{round(time.time())}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(experiment_path, "parameters.json"), "w") as f:
        json.dump(params, f, sort_keys=True, indent=4)

    return output_dir
