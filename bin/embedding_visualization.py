import argparse
import numpy as np
import duckdb
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from evaluating_trajectories.dataset.preprocessing import read_glove_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", default="vocab.txt", type=str)
    parser.add_argument("--vectors_file", default="vectors.txt", type=str)
    parser.add_argument("--data_parquet", default="2019-Oct.parquet", type=str)
    args = parser.parse_args()

    embeddings, embeddings_norm, vocab, ivocab = read_glove_data(
        args.vocab_file, args.vectors_file
    )

    print("scaling embeddings")
    scaled_embs = StandardScaler().fit_transform(embeddings)
    reducer = umap.UMAP()

    print("reducing to 2d")
    embs_2d = reducer.fit_transform(scaled_embs)

    product_df = duckdb.sql(
        f"SELECT DISTINCT ON(product_id) * FROM '{args.data_parquet}'"
    ).df()
    product_df["x"] = None
    product_df["y"] = None

    for prod_id, pos in tqdm(vocab.items()):
        x, y = embs_2d[pos]
        idx = product_df.index[product_df.product_id == int(prod_id)]
        if len(idx) > 0:
            product_df.at[idx[0], "x"] = x
            product_df.at[idx[0], "y"] = y

    fig = px.scatter(product_df, x="x", y="y", color="category_code")
    fig.show()
