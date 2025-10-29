import json

import duckdb
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import evaluating_trajectories.baselines.abid_and_zou_2018 as betacv
import evaluating_trajectories.models.lstm as lstm


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.sequences = [torch.tensor(traj, dtype=torch.long) for traj in trajectories if len(traj) >= 2]

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
            token_id = int(parts[0])
            if token_id in id2idx:
                embeddings[id2idx[token_id]] = np.array(parts[1:], dtype=np.float32)
    return torch.tensor(embeddings, dtype=torch.float32)


class ContinuousCollator:
    """
    Pads ID sequences, then looks up an embedding to return (B,T,D) + mask (B,T).
    If freeze_lookup=True, lookup is wrapped in no_grad().
    Handles both latent representations and glove embeddings for originals.
    """

    def __init__(self, embedding: nn.Embedding, device: torch.device, freeze_lookup: bool = True):
        self.embedding = embedding
        self.device = device
        self.freeze_lookup = freeze_lookup

    def __call__(self, batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        padded_ids = pad_sequence(batch, batch_first=True, padding_value=0)  # (B,T_ids)
        mask = padded_ids != 0  # (B,T)
        if self.freeze_lookup:
            with torch.no_grad():
                x = self.embedding(padded_ids.to(self.device))  # (B,T,D)
        else:
            x = self.embedding(padded_ids.to(self.device))  # (B,T,D)
        return x, mask.to(self.device)


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
    dataset: TrajectoryDataset, collate_fn: ContinuousCollator, encoder: lstm.EncoderRNN, batch_size: int = 64
) -> list[torch.Tensor]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    encoder.eval()
    out_trajs: list[torch.Tensor] = []
    for x, mask in dl:
        enc_outs, _ = encoder(x)  # (B,T,H)
        lengths = mask.sum(dim=1).tolist()
        for b, L in enumerate(lengths):
            out_trajs.append(enc_outs[b, : int(L)].detach())
    return out_trajs


class EncoderForAutowarp(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.base = base_encoder

    def forward(self, x):
        _, h = self.base(x)  # keep only latent
        return h


def train(
    encoder: lstm.EncoderRNN,
    decoder: lstm.AttnDecoderRNN,
    dataloader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
):
    encoder.train()
    decoder.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for _ in range(num_epochs):
        for x, mask in dataloader:  # x: (B,T,D) continuous; mask: (B,T)
            T = x.size(1)

            opt.zero_grad()
            enc_outs, h = encoder(x)
            x_hat = decoder(enc_outs, h, out_len=T)
            loss = masked_mse(x_hat, x, mask)
            loss.backward()
            opt.step()


if __name__ == "__main__":
    TRAJECTORIES_PARQUET = "data/trajectories.parquet"
    MIN_TRAJECTORY_LENGTH = 3
    MAX_TRAJECTORY_LENGTH = 16
    NUM_TRAJECTORIES = 100
    EMBEDDING_MODE = "latent"  # "latent" or "glove"!!!
    HIDDEN = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    con = duckdb.connect()
    query = f"""
        SELECT trajectory
        FROM read_parquet('{TRAJECTORIES_PARQUET}')
        WHERE array_length(trajectory) BETWEEN {MIN_TRAJECTORY_LENGTH} AND {MAX_TRAJECTORY_LENGTH}
        ORDER BY RANDOM()
        LIMIT {NUM_TRAJECTORIES}
    """
    trajectory_list = con.execute(query).fetchall()
    trajectory_list = [row[0] for row in trajectory_list]
    print(f"Loaded {len(trajectory_list)} filtered trajectories")

    # map tokens to indices (pad=0)
    all_ids = set(x for traj in trajectory_list for x in traj)
    id2idx = {nid: idx + 1 for idx, nid in enumerate(sorted(all_ids))}
    vocab_size = len(id2idx) + 1
    trajectory_list_remap = [[id2idx[x] for x in traj] for traj in trajectory_list]
    del trajectory_list

    assert all(1 <= x < vocab_size for traj in trajectory_list_remap for x in traj), "Indexing error!"

    # FROZEN embedding to get continuous states for both encoder and original trajectories
    STATE_DIM = 100  # D; often match hidden size

    dataset = TrajectoryDataset(trajectory_list_remap)

    if EMBEDDING_MODE == "glove":
        embedding_matrix = load_glove_embeddings("glove_vectors.txt", vocab_size, STATE_DIM, id2idx)
        fixed_embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0).to(device)
        collate = ContinuousCollator(fixed_embed, device, freeze_lookup=True)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

        encoder = lstm.EncoderRNN(input_dim=STATE_DIM, hidden_size=HIDDEN).to(device)
        decoder = lstm.AttnDecoderRNN(hidden_size=HIDDEN, output_dim=STATE_DIM).to(device)

        train(encoder, decoder, dataloader, num_epochs=100)

        # freeze before AutoWarp (encoder only; embed is already frozen)
        freeze_before_autowarp(encoder)

        originals_builder = lambda: build_originals_glove(dataset, fixed_embed.weight)

    elif EMBEDDING_MODE == "latent":
        trainable_embed = nn.Embedding(vocab_size, STATE_DIM, padding_idx=0).to(device)
        nn.init.normal_(trainable_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            trainable_embed.weight[0].zero_()

        # TRAIN with gradients flowing through the embedding
        collate = ContinuousCollator(trainable_embed, device, freeze_lookup=False)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

        encoder = lstm.EncoderRNN(input_dim=STATE_DIM, hidden_size=HIDDEN).to(device)
        decoder = lstm.AttnDecoderRNN(hidden_size=HIDDEN, output_dim=STATE_DIM).to(device)

        train(encoder, decoder, dataloader, num_epochs=100)

        # freeze encoder + embedding before AutoWarp
        freeze_before_autowarp(encoder, trainable_embed)

        originals_builder = lambda: build_originals_encoder_outputs(dataset, collate, encoder, batch_size=64)

    else:
        raise ValueError("embedding_mode must be 'glove' or 'latent'.")

    beta_val, alpha, gamma, epsilon = betacv.batched_autowarp(
        dataloader=dataloader,
        encoder=EncoderForAutowarp(encoder),
        originals_builder=originals_builder,
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
