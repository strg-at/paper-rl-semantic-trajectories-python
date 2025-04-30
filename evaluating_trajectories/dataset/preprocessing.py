import pickle
from dataclasses import dataclass
from typing import Any, Generator

import duckdb
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from evaluating_trajectories.utils import consts


@dataclass
class Trajectory:
    nodes: list[int]
    user_session_id: str


@dataclass
class EmbeddingsWithVocab:
    embeddings: npt.NDArray[np.floating]
    embeddings_norm: npt.NDArray[np.floating]
    mask_embedding: npt.NDArray[np.floating]
    vocab: dict[str, int]
    ivocab: dict[int, str]


def load_glove_embeddings(vocab_file: str, vectors_file: str) -> EmbeddingsWithVocab:
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

    mask_embedding = np.zeros(W.shape[-1])
    emb_vocab = EmbeddingsWithVocab(
        embeddings=W, embeddings_norm=W_norm, mask_embedding=mask_embedding, vocab=vocab, ivocab=ivocab
    )
    return emb_vocab


def load_sentencetransf_embeddings(embeddings_file: str, vocab_file: str) -> EmbeddingsWithVocab:
    embs = np.load(embeddings_file)
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    ivocab = {v: k for k, v in vocab.items()}

    model = SentenceTransformer(consts.SENTENCE_TRANSFORMER_MODEL_NAME)
    mask_embedding = model.encode(model.tokenizer.special_tokens_map["mask_token"])

    # Notice, we could have normalized the embeddings setting `normalize_embeddings=True` in `encode`. This keeps
    # consistency with the glove function
    emb_norm = np.zeros(embs.shape)
    d = np.sum(embs**2, 1) ** (0.5)
    emb_norm = (embs.T / d).T

    emb_vocab = EmbeddingsWithVocab(
        embeddings=embs, embeddings_norm=emb_norm, mask_embedding=mask_embedding, vocab=vocab, ivocab=ivocab
    )
    return emb_vocab


def user_sessions(parquet_file: str, column_name: str, delim: str = " ") -> duckdb.DuckDBPyRelation:
    return duckdb.sql(f"SELECT user_session, list({column_name}) AS value FROM '{parquet_file}' GROUP BY user_session")


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
        sessions = conn.sql(f"SELECT user_session, trajectory FROM '{trajectory_file}'")
        while data := sessions.fetchmany(batch_size):
            for session, traj in data:
                yield session, traj
