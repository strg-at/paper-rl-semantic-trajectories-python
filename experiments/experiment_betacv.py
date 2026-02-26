import json
import os
from typing import Callable

import dotenv
import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import rl_semantic_trajectories.baselines.abid_and_zou_2018 as betacv
import rl_semantic_trajectories.dataset.preprocessing as preproc
import rl_semantic_trajectories.models.lstm as lstm

dotenv.load_dotenv()


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.sequences = [
            torch.tensor(traj, dtype=torch.long)
            for traj in tqdm(trajectories, desc="Converting trajectories to tensors")
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def load_glove_embeddings(path: str, vocab_size: int, embedding_dim: int, id2idx: dict) -> torch.Tensor:
    embeddings = np.zeros((vocab_size, embedding_dim))
    embeddings[0] = 0.0  # padding stays zero

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue
            token_id = parts[0]
            if token_id in id2idx:
                embeddings[id2idx[token_id]] = np.array(parts[1:], dtype=np.float32)
    return torch.tensor(embeddings, dtype=torch.float32)


def apply_padding(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    padded_ids = pad_sequence(batch, batch_first=True, padding_value=0)  # (B,T_ids)
    mask = padded_ids != 0  # (B,T)
    return padded_ids, mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target: (B,T,D), mask: (B,T) True for valid steps
    diff2 = (pred - target).pow(2).sum(dim=2)  # (B,T)
    diff2 = diff2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff2.sum() / denom


@torch.no_grad()
def freeze_before_autowarp(encoder: nn.Module, embedding: nn.Embedding | None = None):
    # eval disables dropout etc.
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    if embedding is not None:
        embedding.eval()
        for p in embedding.parameters():
            p.requires_grad_(False)


@torch.no_grad()
def build_originals_glove(dataset: TrajectoryDataset, embedding_weight: torch.Tensor) -> list[torch.Tensor]:
    trajectories = []
    device = embedding_weight.device
    for seq in dataset.sequences:
        ids = seq.to(device)
        trajectories.append(embedding_weight.index_select(0, ids))
    return trajectories


@torch.no_grad()
def build_originals_encoder_outputs(
    dataset: TrajectoryDataset,
    collate_fn: Callable,
    embedding_layer: nn.Embedding,
    encoder: lstm.EncoderRNN,
    batch_size: int = 64,
) -> list[torch.Tensor]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    encoder.eval()
    out_trajs: list[torch.Tensor] = []
    for x, mask in dl:
        x = x.to(device)
        mask = mask.to(device)
        enc_outs, _ = encoder(embedding_layer(x))  # (B,T,H)
        lengths = mask.sum(dim=1).tolist()
        for b, L in enumerate(lengths):
            out_trajs.append(enc_outs[b, : int(L)].detach())
    return out_trajs


def train(
    embedding_layer: nn.Embedding,
    encoder: lstm.EncoderRNN,
    decoder: lstm.AttnDecoderRNN,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    min_delta: float = 1e-4,
):
    encoder.train()
    decoder.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    # Early stopping variables
    best_loss = float("inf")
    patience_counter = 0

    pbar = tqdm(range(num_epochs), desc="Training LSTM encoder.", position=0)
    for epoch in pbar:
        losses = []
        batch_pbar = tqdm(dataloader, desc="Training batches", position=1, leave=False)
        for x, mask in batch_pbar:  # x: (B,T,D) continuous; mask: (B,T)
            T = x.size(1)
            x = x.to(device)
            mask = mask.to(device)
            opt.zero_grad()
            x = embedding_layer(x)
            enc_outs, h = encoder(x)
            x_hat = decoder(enc_outs, h, out_len=T)
            loss = masked_mse(x_hat, x, mask)
            loss.backward()
            losses.append(loss.item())
            opt.step()
            batch_pbar.set_description(f"Training batches, avg Loss: {np.mean(losses):.4f}")

        epoch_loss = np.mean(losses)
        pbar.set_description(f"Training LSTM encoder. Avg Epoch Loss: {epoch_loss:.4f}")

        # Early stopping check
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
            break


if __name__ == "__main__":
    TRAJECTORIES_FILE = os.getenv("TRAJECTORIES_FILE", "data/trajectories.parquet")
    MIN_TRAJECTORY_LENGTH = int(os.getenv("MIN_TRAJECTORY_LENGTH", 3))
    MAX_TRAJECTORY_LENGTH = int(os.getenv("MAX_TRAJECTORY_LENGTH", 16))
    PERCENTAGE_TRAJECTORIES_SAMPLE = float(os.getenv("PERCENTAGE_TRAJECTORIES_SAMPLE", 0.3))
    EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "glove")  # "latent" or "glove"!!!
    GLOVE_VECTORS_FILE = os.getenv("GLOVE_VECTORS_FILE", "glove_vectors.txt")
    GLOVE_VOCAB_FILE = os.getenv("GLOVE_VOCAB_FILE", "glove_vocab.txt")
    HIDDEN = int(os.getenv("HIDDEN_SIZE", 100))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    con = duckdb.connect()
    sample_size = con.execute(  # pyright: ignore[reportOptionalSubscript]
        f"SELECT round(count(*) * {PERCENTAGE_TRAJECTORIES_SAMPLE}) FROM read_parquet('{TRAJECTORIES_FILE}')"
    ).fetchone()[0]
    query = f"""
        SELECT user_session, trajectory
        FROM read_parquet('{TRAJECTORIES_FILE}')
        WHERE array_length(trajectory) BETWEEN {MIN_TRAJECTORY_LENGTH} AND {MAX_TRAJECTORY_LENGTH}
        USING SAMPLE {sample_size} ROWS
    """
    print("Loading data...")
    trajectory_dict = dict(con.execute(query).fetchall())
    trajectory_list = list(trajectory_dict.values())
    print(f"Loaded {len(trajectory_list)} filtered trajectories")

    embs_with_vocab = preproc.load_glove_embeddings(GLOVE_VOCAB_FILE, GLOVE_VECTORS_FILE)
    vocab = {int(k): v for k, v in embs_with_vocab.vocab.items()}
    vocab_size = len(vocab)

    # Based on the desired configuration, GloVe might have discarded some nodes. We remove trajectories that contain those nodes.
    # These trajectories are removed in all cases to make things comparable.
    trajectory_list_remap = [[vocab.get(x) for x in traj] for traj in tqdm(trajectory_list)]
    trajectory_list_remap = [traj for traj in trajectory_list_remap if all(t is not None for t in traj)]
    del trajectory_list

    # FROZEN embedding to get continuous states for both encoder and original trajectories
    STATE_DIM = 100  # D; often match hidden size

    dataset = TrajectoryDataset(trajectory_list_remap)

    if EMBEDDING_MODE == "glove":
        emb_layer = nn.Embedding.from_pretrained(
            torch.tensor(embs_with_vocab.embeddings, dtype=torch.float32), freeze=True, padding_idx=0
        ).to(device)
        dataloader = DataLoader(dataset, batch_size=768, shuffle=True, collate_fn=apply_padding, num_workers=10)

        encoder = lstm.EncoderRNN(input_dim=STATE_DIM, hidden_size=HIDDEN).to(device)
        decoder = lstm.AttnDecoderRNN(hidden_size=HIDDEN, output_dim=STATE_DIM).to(device)

        train(emb_layer, encoder, decoder, dataloader, device=device, num_epochs=100)

        # freeze before AutoWarp (encoder only; embed is already frozen)
        freeze_before_autowarp(encoder)

        originals_builder = lambda: build_originals_glove(dataset, emb_layer.weight)

    elif EMBEDDING_MODE == "latent":
        emb_layer = nn.Embedding(vocab_size, STATE_DIM, padding_idx=0).to(device)

        # TRAIN with gradients flowing through the embedding
        dataloader = DataLoader(dataset, batch_size=768, shuffle=True, collate_fn=apply_padding, num_workers=10)

        encoder = lstm.EncoderRNN(input_dim=STATE_DIM, hidden_size=HIDDEN).to(device)
        decoder = lstm.AttnDecoderRNN(hidden_size=HIDDEN, output_dim=STATE_DIM).to(device)

        train(emb_layer, encoder, decoder, dataloader, device=device, num_epochs=100)

        # freeze encoder + embedding before AutoWarp
        freeze_before_autowarp(encoder, emb_layer)

        originals_builder = lambda: build_originals_encoder_outputs(
            dataset, apply_padding, emb_layer, encoder, batch_size=64
        )

    else:
        raise ValueError("embedding_mode must be 'glove' or 'latent'.")

    # the Autowarp algorithm needs to compute a distance matrix between all trajectories.
    # This does not scale well at all. Our solution for now is to:
    #   1. Limit the amount of trajectory we deal with;
    #   2. Use duckdb to compute the euclidean distance, and store the distance matrix into a parquet file;
    #   3. Use duckdb to compute the approx. quantile and do the sampling, reading from the parquet file.
    con.execute(f"CREATE TABLE trajs (id BIGINT, trajectory float[{HIDDEN}])")
    print("Computing latent representations...")
    latents = betacv.compute_latents(dataloader, emb_layer, betacv.EncoderForAutowarp(encoder))
    print(f"Computed {latents.size(0)} latent representations of dimension {latents.size(1)}")

    print("Storing distance matrix into duckdb...")
    # Use a df to insert data, this makes it much much faster
    df = pd.DataFrame({"id": range(len(latents)), "trajectory": latents.detach().cpu().tolist()})
    con.execute("INSERT INTO trajs FROM df")
    con.execute("SET preserve_insertion_order = false;")
    con.execute("COPY trajs TO 'latenttrajs.parquet'")

    con.sql(
        "CREATE VIEW dist AS SELECT a.id id_1, b.id id_2, array_distance(a.trajectory, b.trajectory) AS DIST FROM trajs a CROSS JOIN trajs b WHERE a.id < b.id"
    )
    print("Saving distance matrix onto parquet file...")
    con.sql("COPY dist TO 'distance_matrix.parquet'")

    beta_val, alpha, gamma, epsilon = betacv.batched_autowarp(
        conn=con,
        distparquet_path="distance_matrix.parquet",
        latents=latents,
        originals_builder=originals_builder,
        device=device,
        pair_batch_size=2048,
        max_iters=100,
    )
    with open("autowarp_results_latent.json", "w") as f:
        json.dump(
            {
                "betaCV": beta_val,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
            },
            f,
            indent=2,
        )
    with open("autowarp_training_ids.json", "w") as f:
        ids = list(trajectory_dict.keys())
        json.dump(ids, f, indent=2)
