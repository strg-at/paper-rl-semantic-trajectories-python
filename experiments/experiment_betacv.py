import json
import duckdb
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List

import evaluating_trajectories.baselines.abid_and_zou_2018 as betacv
import evaluating_trajectories.models.lstm as lstm


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_len: int):
        self.sequences = [torch.tensor(traj[:max_len], dtype=torch.long) for traj in trajectories if len(traj) >= 2]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]


class ContinuousCollator:
    """
    Pads ID sequences, then looks up a FROZEN embedding to return continuous (B,T,D) + mask (B,T).
    Padding id=0 maps to the zero vector.
    """
    def __init__(self, embedding: nn.Embedding, device: torch.device, max_len: int):
        self.embedding = embedding
        self.device = device
        self.max_len = max_len

    def __call__(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded_ids = pad_sequence(batch, batch_first=True, padding_value=0)  # (B,T_ids)
        if padded_ids.size(1) > self.max_len:
            padded_ids = padded_ids[:, :self.max_len]
        mask = (padded_ids != 0)  # (B,T)
        with torch.no_grad():     # frozen lookup
            x = self.embedding(padded_ids.to(self.device))  # (B,T,D)
        return x, mask.to(self.device)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target: (B,T,D), mask: (B,T) True for valid steps
    diff2 = (pred - target).pow(2).sum(dim=2)  # (B,T)
    diff2 = diff2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff2.sum() / denom


class EncoderForAutowarp(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.base = base_encoder
    def forward(self, x):
        _, h = self.base(x)   # keep only latent
        return h


def train(
    encoder: lstm.EncoderRNN,
    decoder: lstm.AttnDecoderRNN,
    dataloader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    max_len: int | None = None,
    device: str | torch.device = "cpu",
):
    encoder.train(); decoder.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for _ in range(num_epochs):
        for x, mask in dataloader:      # x: (B,T,D) continuous; mask: (B,T)
            if max_len is not None:
                x = x[:, :max_len]
                mask = mask[:, :max_len]
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
    STATE_DIM = 128  # D; often match hidden size
    fixed_state_embed = nn.Embedding(vocab_size, STATE_DIM, padding_idx=0).to(device)
    with torch.no_grad():
        fixed_state_embed.weight.normal_(mean=0.0, std=0.02)
        fixed_state_embed.weight[0].zero_()           # padding -> exact zeros
    fixed_state_embed.weight.requires_grad_(False)

    dataset = TrajectoryDataset(trajectory_list_remap, max_len=MAX_TRAJECTORY_LENGTH)
    collate = ContinuousCollator(fixed_state_embed, device, max_len=MAX_TRAJECTORY_LENGTH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

    # Check if the first batch has the expected shape
    for x, mask in dataloader:
        assert x.dim() == 3 and x.size(2) == STATE_DIM
        assert mask.dtype == torch.bool
        break

    encoder = lstm.EncoderRNN(input_dim=STATE_DIM, hidden_size=128).to(device)
    decoder = lstm.AttnDecoderRNN(hidden_size=128, output_dim=STATE_DIM).to(device)

    # train encoder with masked MSE on continuous sequences
    train(encoder, decoder, dataloader, num_epochs=100, max_len=MAX_TRAJECTORY_LENGTH, device=device)

    # Run Autowarp (uses SAME frozen embedding for originals; and the encoder for latents)
    beta_val, alpha, gamma, epsilon = betacv.batched_autowarp(
        dataloader=dataloader,
        dataset=dataset,
        encoder=EncoderForAutowarp(encoder),
        max_len=MAX_TRAJECTORY_LENGTH,
        max_iters=100,
        embedding_weight=fixed_state_embed.weight,   # same frozen table
    )

    with open("autowarp_result.json", "w") as f:
        json.dump({
            "beta": beta_val,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon
        }, f, indent=2)

    print("DONE")
