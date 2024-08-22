from evaluating_trajectories.dataset import preprocessing
from tqdm import tqdm
import duckdb
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to the parquet file", default="2019-Oct.parquet"
    )
    parser.add_argument("--output-file", type=str, help="Output file", required=True)
    parser.add_argument(
        "--min-trajectory-length", type=int, help="Minimum trajectory length", default=3
    )
    args = parser.parse_args()
    count = (
        duckdb.sql(f"SELECT DISTINCT user_session FROM '{args.path}'")
        .count("*")
        .fetchone()[0]
    )
    batch_size = 7_000_000  # this should be ok if you have ~32gb of ram
    user_sessions_it = preprocessing.user_sessions_iterator(
        args.path, ["product_id"], batch_size=batch_size
    )
    pbar = tqdm(desc="Writing trajectories in glove format", total=count // batch_size)
    with open(args.output_file, "w") as f:
        for trajectory_groups in user_sessions_it:
            trajectory_groups = map(lambda t: t[1], trajectory_groups)
            trajectory_groups = filter(
                lambda t: len(t) >= args.min_trajectory_length, trajectory_groups
            )
            str_products = map(
                lambda t: " ".join(map(lambda p: str(p["product_id"]), t)) + "\n",
                trajectory_groups,
            )

            f.writelines(str_products)
            pbar.update()
