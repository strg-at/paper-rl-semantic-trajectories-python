from evaluating_trajectories.dataset.preprocessing import load_glove_embeddings
import argparse
import numpy as np
import duckdb
import random
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", default="vocab.txt", type=str)
    parser.add_argument("--vectors_file", default="vectors.txt", type=str)
    parser.add_argument("--data_parquet", default="2019-Oct.parquet", type=str)
    args = parser.parse_args()

    emb_with_vocab = load_glove_embeddings(args.vocab_file, args.vectors_file)
    embeddings_norm = emb_with_vocab.embeddings_norm
    vocab = emb_with_vocab.vocab
    ivocab = emb_with_vocab.ivocab

    rand_prods = duckdb.sql(
        f"SELECT product_id, brand, category_code FROM '{args.data_parquet}' WHERE brand is not NULL and category_code is not NULL USING SAMPLE 5"
    ).fetchall()
    prod_dicts = []
    for prod in rand_prods:
        prod_id, brand, category_code = prod
        prod_vec = emb_with_vocab.embeddings_norm[vocab[str(prod_id)]]
        d = np.sum(prod_vec**2) ** 0.5
        vec_norm = (prod_vec.T / d).T

        dist = np.dot(embeddings_norm, vec_norm.T)
        dist[vocab[str(prod_id)]] = -np.inf  # we don't want to show this product

        asorted = np.argsort(-dist)[:5]
        prod_details_dict = {
            "prod_id": prod_id,
            "brand": brand,
            "category_code": category_code,
            "similar": [],
        }
        for idx in tqdm(asorted):
            category_code = duckdb.sql(
                f"SELECT category_code FROM '{args.data_parquet}' WHERE product_id = {ivocab[idx]}"
            ).fetchone()
            brand = duckdb.sql(f"SELECT brand FROM '{args.data_parquet}' WHERE product_id = {ivocab[idx]}").fetchone()
            if category_code is None or brand is None:
                continue
            prod_details_dict["similar"].append(
                {
                    "prod_id": ivocab[idx],
                    "distance": dist[idx].item(),
                    "category_code": category_code[0],
                    "brand": brand[0],
                }
            )
        prod_dicts.append(prod_details_dict)
