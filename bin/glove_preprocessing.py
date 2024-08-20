from evaluating_trajectories.dataset import preprocessing
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to the parquet file", default="2019-Oct.parquet"
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Column to group by",
        default="product_id",
    )
    parser.add_argument("--output-file", type=str, help="Output file", required=True)
    args = parser.parse_args()
    user_sessions = preprocessing.user_sessions(args.path, args.column)
    product_ids = user_sessions["value"]
    count = product_ids.count("value").fetchone()[0] // 1000
    pbar = tqdm(desc="Writing trajectories in glove format", total=count)
    with open(args.output_file, "a") as f:
        while products := product_ids.fetchmany(1000):
            str_products = map(
                lambda t: " ".join(str(ti) for ti in t[0]) + "\n", products
            )
            f.writelines(str_products)
            pbar.update()
