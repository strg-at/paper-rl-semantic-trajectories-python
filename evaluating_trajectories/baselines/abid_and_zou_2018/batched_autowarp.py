import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import evaluating_trajectories.models.lstm as lstm
import experiments.experiment_betacv as exp_betacv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_latents_for_threshold(dataloader: DataLoader, encoder: exp_betacv.EncoderForAutowarp, max_len: int) -> torch.Tensor:
    """
    Works with your discrete-ID encoder (nn.Embedding inside).
    Returns H: (N, H) final hidden state per trajectory (no grad).
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for x, _ in dataloader: # x: (B, T, D)
            x = x[:, :max_len].to(device)
            h = encoder(x) # h: (B, H)  (wrapper returns latent directly)
            chunks.append(h)
    return torch.cat(chunks, dim=0) 


def _sigma_eps(x, eps_over_1_minus):
    # x >= 0 scalar tensor, y = eps/(1-eps)
    return eps_over_1_minus * torch.tanh(x / (eps_over_1_minus + 1e-12))

def warping_distance(X, Y, raw_alpha, raw_gamma, raw_epsilon):
    """
    X: (n, D), Y: (m, D) continuous trajectories
    Returns scalar distance with Eq. (1) exactly:
      diag:        σ(||X[i]-Y[j]||, ε/(1-ε))
      horiz/vert:  (α/(1-α))*σ(||X[i]-Y[j]||, ε/(1-ε)) + γ
    """
    device = X.device
    n, m = X.size(0), Y.size(0)

    alpha    = torch.sigmoid(raw_alpha)     # (0,1)
    gamma    = F.softplus(raw_gamma)        # >= 0
    epsilon  = torch.sigmoid(raw_epsilon)   # (0,1)
    a_scaled = alpha / (1 - alpha + 1e-12)
    eps_par  = epsilon / (1 - epsilon + 1e-12)

    # DP table
    dp = torch.full((n+1, m+1), float('inf'), device=device, dtype=X.dtype)
    dp[0, 0] = X.new_tensor(0.0)

    null = torch.zeros(X.size(1), device=device)

    for i in range(0, n+1):
        for j in range(0, m+1):
            if i == 0 and j == 0: 
                continue
            # End-state vectors (use null when index == 0)
            xi = X[i-1] if i>0 else null
            yj = Y[j-1] if j>0 else null
            dist = torch.norm(xi - yj)
            s = _sigma_eps(dist, eps_par)

            candidates = []
            # diagonal: (i-1, j-1) -> (i, j)
            if i>0 and j>0:
                candidates.append(dp[i-1, j-1] + s)
            # vertical: (i-1, j) -> (i, j)
            if i>0:
                candidates.append(dp[i-1, j] + a_scaled*s + gamma)
            # horizontal: (i, j-1) -> (i, j)
            if j>0:
                candidates.append(dp[i, j-1] + a_scaled*s + gamma)

            dp[i, j] = torch.min(torch.stack(candidates))
    return dp[n, m]


def distance_matrix(H: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances between latent representations"""
    return torch.cdist(H, H)  # (N, N)


def precompute_pairs(D_lat: torch.Tensor, percentile: int):
    """Extract pairs for close/all sampling based on latent distances"""
    iu = torch.triu_indices(D_lat.size(0), D_lat.size(1), offset=1)
    tri = D_lat[iu[0], iu[1]]
    delta = torch.quantile(tri, percentile/100.0)
    
    # Close pairs: latent distance < threshold
    close_mask = D_lat < delta
    Ci, Cj = torch.where(torch.triu(close_mask, diagonal=1))
    
    # All pairs
    Ai, Aj = iu[0], iu[1]
    
    if Ci.numel() == 0:
        raise RuntimeError(f"No close pairs found with percentile {percentile}. Try increasing percentile or check latent representations.")
    
    return (Ci, Cj, Ai, Aj, delta)


def build_originals(dataset: exp_betacv.TrajectoryDataset, max_len: int, embedding_weight: torch.Tensor) -> list[torch.Tensor]:
    """
    Convert discrete token sequences to continuous trajectories via embedding lookup.
    embedding_weight: (vocab_size, D) tensor with frozen weights
    Returns list of (Li, D) tensors
    """
    trajectories = []
    device = embedding_weight.device
    
    for seq in dataset.sequences:
        ids = seq[:max_len].to(device)
        continuous_traj = embedding_weight.index_select(0, ids)
        trajectories.append(continuous_traj)
    
    return trajectories


def sum_over_pairs(trajectories: list[torch.Tensor], alpha, gamma, epsilon, pairs: list[tuple[int,int]]) -> torch.Tensor:
    """Compute sum of warping distances over given pairs"""
    total = torch.zeros((), device=DEVICE)
    
    for i, j in pairs:
        traj_A, traj_B = trajectories[i], trajectories[j]
        
        # Skip very short trajectories
        if traj_A.size(0) < 2 or traj_B.size(0) < 2:
            continue
            
        dist = warping_distance(traj_A, traj_B, alpha, gamma, epsilon)
        
        if torch.isfinite(dist):
            total = total + dist
    
    return total


def batched_autowarp(
    dataloader: DataLoader,
    encoder: exp_betacv.EncoderForAutowarp,
    dataset: exp_betacv.TrajectoryDataset,
    max_len: int,
    percentile: int = 20, # this needs to be low, so close pairs are only pairs below the 20th percentile
    lr: float = 0.01,
    pair_batch_size: int = 64,
    max_iters: int = 50,
    convergence_tol: float = 1e-4,
    embedding_weight: torch.Tensor | None = None,
):
    """
    Batched Autowarp (single run, no restarts):
      1) get latents h from encoder
      2) define δ from pairwise latent distances
      3) minimize betaCV over raw α, γ, ε with correct transforms
    """
    device = next(encoder.parameters()).device
    print(f"Running Autowarp on device: {device}")

    # Step 1: latents H
    print("Computing latent representations...")
    H = compute_latents_for_threshold(dataloader, encoder, max_len)  # (N, H)
    print(f"Computed {H.size(0)} latent representations of dimension {H.size(1)}")

    # Step 2: originals (continuous) via frozen embedding
    if embedding_weight is None:
        raise ValueError("embedding_weight is required to convert discrete tokens to continuous trajectories.")
    trajectories = build_originals(dataset, max_len=max_len, embedding_weight=embedding_weight)

    D_latent = distance_matrix(H)
    Ci, Cj, Ai, Aj, delta = precompute_pairs(D_latent, percentile)
    print(f"Found {len(Ci)} close pairs and {len(Ai)} total pairs (threshold: {float(delta):.4f})")

    # randomly initialize raw parameters
    raw_alpha   = torch.randn((), device=device, requires_grad=True)
    raw_gamma   = torch.randn((), device=device, requires_grad=True)
    raw_epsilon = torch.randn((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_alpha, raw_gamma, raw_epsilon], lr=lr)

    print("Starting parameter optimization...")
    prev_beta = None
    current_beta = None

    for step in range(max_iters):
        # Sample S pairs from close and S from all
        S_close = min(pair_batch_size, Ci.numel())
        S_all   = min(pair_batch_size, Ai.numel())
        idx_c = torch.randint(0, Ci.numel(), (S_close,), device=device)
        idx_a = torch.randint(0, Ai.numel(), (S_all,),   device=device)
        pairs_close = [(int(Ci[k]), int(Cj[k])) for k in idx_c]
        pairs_all   = [(int(Ai[k]), int(Aj[k])) for k in idx_a]

        optimizer.zero_grad()
        num = sum_over_pairs(trajectories, raw_alpha, raw_gamma, raw_epsilon, pairs_close)
        den = sum_over_pairs(trajectories, raw_alpha, raw_gamma, raw_epsilon, pairs_all)
        beta_hat = num / (den + 1e-12)
        beta_hat.backward()
        optimizer.step()

        current_beta = float(beta_hat.detach().cpu())
        if prev_beta is not None and abs(current_beta - prev_beta) < convergence_tol:
            print(f"Converged at step {step} with β̂ = {current_beta:.4f}")
            break
        prev_beta = current_beta

        if step % 10 == 0:
            a = torch.sigmoid(raw_alpha).item()
            g = F.softplus(raw_gamma).item()
            e = torch.sigmoid(raw_epsilon).item()
            print(f"Step {step:3d} | β̂: {current_beta:.4f} | α: {a:.3f} | γ: {g:.3f} | ε: {e:.3f}")

    # Final parameter values
    final_alpha   = torch.sigmoid(raw_alpha).item()
    final_gamma   = F.softplus(raw_gamma).item()
    final_epsilon = torch.sigmoid(raw_epsilon).item()
    final_beta    = current_beta if current_beta is not None else float('nan')

    print("\nFinal Results:")
    print(f"β̂: {final_beta:.4f}")
    print(f"α: {final_alpha:.3f}")
    print(f"γ: {final_gamma:.3f}")
    print(f"ε: {final_epsilon:.3f}")

    return final_beta, final_alpha, final_gamma, final_epsilon
