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
    count = (
        duckdb.sql(f"SELECT DISTINCT user_session FROM '{args.path}'")
        .count("*")
        .fetchone()[0]
    )
    batch_size = args.batch_size
    user_sessions_it = preprocessing.user_sessions_iterator(
        args.path, ["product_id"], batch_size=batch_size
    )
    pbar = tqdm(desc="Writing trajectories in glove format", total=count // batch_size)
    with open(args.output_file, "w") as f:
        for _, trajectory in user_sessions_it:
            str_products = " ".join(map(str, trajectory))
            f.write(str_products)
            pbar.update()
