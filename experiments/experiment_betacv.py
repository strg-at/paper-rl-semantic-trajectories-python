import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from evaluating_trajectories.baselines.abid_and_zou_2018 import batched_autowarp
from evaluating_trajectories.models.lstm import DecoderRNN, EncoderRNN


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_len):
        self.sequences = [torch.tensor(traj[:max_len], dtype=torch.long) for traj in trajectories if len(traj) >= 2]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch):
    padded = pad_sequence(batch, batch_first=True, padding_value=0)  # 0 is padding
    inputs = padded[:, :-1].to(device)
    targets = padded[:, 1:].to(device)
    return inputs.to(device), targets.to(device)
    

def train(
    encoder, decoder,
    trajectories: list,
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    max_len=10
):
    dataset = TrajectoryDataset(trajectories, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    encoder.train()
    decoder.train()

    criterion = nn.NLLLoss(ignore_index=0)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(inputs)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor=targets)
            targets = targets.clamp(0, vocab_size-1)
            if targets.max() >= vocab_size or targets.min() < 0:
                raise ValueError("Target out of bounds")
            # Flatten for loss
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                targets.view(-1)
            )

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

def collate_fn(batch):
    # batch is a list of tensors (each of variable length)
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    inputs = padded[:, :-1]  # everything except last token
    targets = padded[:, 1:]  # everything except first token (next-step prediction)
    return inputs.to(device), targets.to(device)



num_trajectories = 10000
MAX_TRAJECTORY_LENGTH = 52
trajectories_file = "data/trajectories.parquet"

df = pd.read_parquet(trajectories_file, columns=["trajectory"])
subset = df.sample(n=num_trajectories, random_state=42)
trajectory_list = subset["trajectory"].tolist()
del df, subset

# map nodes to dense indices to reduce memory consumption
all_ids = set(x for traj in trajectory_list for x in traj)
id2idx = {nid: idx + 1 for idx, nid in enumerate(sorted(all_ids))}   # 0 is pad
trajectory_list_remap = [[id2idx[x] for x in traj] for traj in trajectory_list]
vocab_size = len(id2idx) + 1  # +1 for padding

for i, traj in enumerate(trajectory_list_remap):
    for x in traj:
        if not (1 <= x < vocab_size):
            print(f"ERROR: Trajectory {i} contains out-of-range token {x} (vocab_size={vocab_size})")
assert all(1 <= x < vocab_size for traj in trajectory_list_remap for x in traj), "Indexing error!"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TrajectoryDataset(trajectory_list_remap, max_len=MAX_TRAJECTORY_LENGTH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

for inputs, targets in dataloader:
    assert inputs.max() < vocab_size
    assert inputs.min() >= 0
    assert targets.max() < vocab_size
    assert targets.min() >= 0
    break

# 5. Model init (embedding uses padding_idx=0)
encoder = EncoderRNN(input_size=vocab_size, hidden_size=128).to(device)
decoder = DecoderRNN(hidden_size=128, output_size=vocab_size, max_trajectory_length=MAX_TRAJECTORY_LENGTH).to(device)

# Train using remapped dataset
train(encoder, decoder, dataset, num_epochs=100)

# After training: autowarp
beta_val, alpha, gamma, epsilon = batched_autowarp(
    trajectories=dataset,
    encoder=encoder,
    max_len=MAX_TRAJECTORY_LENGTH,
    max_iters=100,
)

# Save results
with open("autowarp_result.json", "w") as f:
    json.dump({
        "beta": beta_val,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon
    }, f, indent=2)

print("DONE")