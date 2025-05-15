from evaluating_trajectories.dataset import preprocessing
from tqdm import tqdm
import duckdb
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the trajectories csv file",
        default="data/trajectories.csv",
    )
    parser.add_argument("--output-file", type=str, help="Output file", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for loading the data. Defaults to 10k.",
        default=10_000,
    )
    args = parser.parse_args()

    duckdb.sql(
        f"""
    COPY (
        -- basically equivalent to " ".join(trajectory) in python
        SELECT list_reduce(apply(trajectory, t -> t::varchar), (acc, t) -> concat(acc, ' ', t))
        FROM '{args.path}'
    ) TO '{args.output_file}' (header false)
    """
    )
