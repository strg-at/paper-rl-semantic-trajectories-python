import duckdb
import numpy as np
from typing import Generator, Any
from dataclasses import dataclass


@dataclass
class Trajectory:
    nodes: list[int]
    user_session_id: str


def read_glove_data(vocab_file, vectors_file):
    with open(vocab_file, "r") as f:
        words = [x.rstrip().split(" ")[0] for x in f.readlines()]
    with open(vectors_file, "r") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(" ")
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == "<unk>":
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = np.sum(W**2, 1) ** (0.5)
    W_norm = (W.T / d).T
    return W, W_norm, vocab, ivocab


def user_sessions(parquet_file: str, column_name: str) -> duckdb.DuckDBPyRelation:
    return duckdb.sql(
        f"SELECT user_session, list({column_name}) AS value FROM '{parquet_file}' GROUP BY user_session"
    )


def user_sessions_iterator(
    trajectory_file: str, columns: list[str], batch_size=10_000
) -> Generator[tuple[str, list[dict[str, Any]]], None, None]:
    """
    Iterator over user sessions, to be used when memory usage is an issue
    """
    if len(columns) == 0:
        columns = duckdb.sql(f"SELECT * FROM '{trajectory_file}' LIMIT 1").columns
    with duckdb.connect() as conn:
        conn.execute("SET preserve_insertion_order = false;")
        sessions = conn.sql("SELECT user_session, trajectory FROM '{parquet_file}'")
        while data := sessions.fetchmany(batch_size):
            for session, traj in data:
                yield session, traj
