import typing

import duckdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from numba import jit, prange

import evaluating_trajectories.models.lstm as lstm


class EncoderForAutowarp(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.base = base_encoder

    def forward(self, x):
        _, h = self.base(x)  # keep only latent
        return h


def compute_latents(dataloader: DataLoader, embedding_layer: nn.Embedding, encoder: EncoderForAutowarp) -> torch.Tensor:
    """
    Works with your discrete-ID encoder (nn.Embedding inside).
    Returns H: (N, H) final hidden state per trajectory (no grad).
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for x, _ in dataloader:  # x: (B, T, D)
            x = x.to(device)
            x = embedding_layer(x)
            h = encoder(x)  # h: (B, H)  (wrapper returns latent directly)
            chunks.append(h)
    return torch.cat(chunks, dim=0)


def warping_distance_torch(X, Y, raw_alpha, raw_gamma, raw_epsilon):
    """
    X: (n, D), Y: (m, D) continuous trajectories
    Returns scalar distance with Eq. (1) exactly:
      diag:        σ(||X[i]-Y[j]||, ε/(1-ε))
      horiz/vert:  (α/(1-α))*σ(||X[i]-Y[j]||, ε/(1-ε)) + γ

    Notice: this function should only be used when running this script, to optimize the three parameters.
        When using it with RL models, the `warping_distance_numba` should be preferred.
    """
    device = X.device
    n, m = X.size(0), Y.size(0)

    alpha = torch.sigmoid(raw_alpha)  # (0,1)
    gamma = F.softplus(raw_gamma)  # >= 0
    epsilon = torch.sigmoid(raw_epsilon)  # (0,1)
    a_scaled = alpha / (1 - alpha + 1e-12)
    eps_par = epsilon / (1 - epsilon + 1e-12)

    # DP table
    dp = torch.full((n + 1, m + 1), float("inf"), device=device, dtype=X.dtype)
    dp[0, 0] = X.new_tensor(0.0)

    null = torch.zeros(X.size(1), device=device)

    for i in range(0, n + 1):
        for j in range(0, m + 1):
            if i == 0 and j == 0:
                continue
            # End-state vectors (use null when index == 0)
            xi = X[i - 1] if i > 0 else null
            yj = Y[j - 1] if j > 0 else null
            dist = torch.norm(xi - yj)
            s = _sigma_eps(dist, eps_par)

            candidates = []
            # diagonal: (i-1, j-1) -> (i, j)
            if i > 0 and j > 0:
                candidates.append(dp[i - 1, j - 1] + s)
            # vertical: (i-1, j) -> (i, j)
            if i > 0:
                candidates.append(dp[i - 1, j] + a_scaled * s + gamma)
            # horizontal: (i, j-1) -> (i, j)
            if j > 0:
                candidates.append(dp[i, j - 1] + a_scaled * s + gamma)

            dp[i, j] = torch.min(torch.stack(candidates))
    return dp[n, m]


def _sigma_eps(x, eps_over_1_minus):
    # x >= 0 scalar tensor, y = eps/(1-eps)
    return eps_over_1_minus * torch.tanh(x / (eps_over_1_minus + 1e-12))


def warping_distance_numba(X, Y, raw_alpha, raw_gamma, raw_epsilon, normalization=True):
    """
    Numba implementation of the warping distance, makes it much faster once the parameters are fixed.

    Notice that we also perform normalization when specified, as we do for the levenshtein distance.
    """
    # 1. Precompute params and cost matrix on GPU (fast)
    device = X.device
    n, D = X.shape
    m = Y.shape[0]

    alpha = torch.sigmoid(raw_alpha)
    gamma = F.softplus(raw_gamma)
    epsilon = torch.sigmoid(raw_epsilon)
    a_scaled = (alpha / (1 - alpha + 1e-12)).item()
    eps_par = (epsilon / (1 - epsilon + 1e-12)).item()
    val_gamma = gamma.item()

    zeros = torch.zeros((1, D), device=device)
    X_pad = torch.cat([zeros, X], dim=0)
    Y_pad = torch.cat([zeros, Y], dim=0)

    # Calculate costs on GPU then move to CPU
    dist_mat = torch.cdist(X_pad, Y_pad).cpu().numpy()

    # 2. Run JIT-compiled DP loop
    return _numba_core(dist_mat, n, m, a_scaled, val_gamma, eps_par, normalization=normalization)


@jit(nopython=True, parallel=True)
def _numba_core(dist_mat, n, m, a_scaled, gamma, eps_par, normalization=True):
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(n + 1):
        for j in prange(m + 1):
            if i == 0 and j == 0:
                continue

            s = eps_par * np.tanh(dist_mat[i, j] / (eps_par + 1e-12))

            v_diag = np.inf
            v_vert = np.inf
            v_horiz = np.inf

            if i > 0 and j > 0:
                v_diag = dp[i - 1, j - 1] + s
            if i > 0:
                v_vert = dp[i - 1, j] + a_scaled * s + gamma
            if j > 0:
                v_horiz = dp[i, j - 1] + a_scaled * s + gamma

            dp[i, j] = min(v_diag, min(v_vert, v_horiz))

    score = dp[n, m]
    if normalization:
        # We take eps_par since it's independent. Also, notice that
        # sigma(d, eps_par) = eps_par * tanh(d/eps_par) approaches eps_par as d -> inf
        max_cost = a_scaled * eps_par + gamma
        max_score = max(n, m) * max_cost
        if max_score > 1e-12:
            score /= max_score
    return score


def sample_close_pairs(
    conn: duckdb.DuckDBPyConnection, parquet_file_path: str, quantile: float, sample_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sample_ids = conn.sql(
        f"SELECT id_1, id_2 FROM (SELECT id_1, id_2 FROM '{parquet_file_path}' WHERE dist <= ?) USING SAMPLE {sample_size}",
        params=(quantile,),
    )
    trajectory_pairs = conn.sql(
        """
        SELECT t1.trajectory as traj_1, t2.trajectory as traj_2
        FROM sample_ids s
        JOIN trajs t1 ON s.id_1 = t1.id
        JOIN trajs t2 ON s.id_2 = t2.id
    """
    ).fetchnumpy()
    m_traj1 = map(torch.tensor, trajectory_pairs["traj_1"])
    m_traj2 = map(torch.tensor, trajectory_pairs["traj_2"])
    pairs = list(zip(m_traj1, m_traj2))
    return pairs


def sample_pairs(
    conn: duckdb.DuckDBPyConnection, parquet_files_path: str, sample_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sample_ids = conn.sql(f"SELECT id_1, id_2 FROM '{parquet_files_path}' USING SAMPLE {sample_size}")
    trajectory_pairs = conn.sql(
        """
        SELECT t1.trajectory as traj_1, t2.trajectory as traj_2
        FROM sample_ids s
        JOIN trajs t1 ON s.id_1 = t1.id
        JOIN trajs t2 ON s.id_2 = t2.id
    """
    ).fetchnumpy()
    m_traj1 = map(torch.tensor, trajectory_pairs["traj_1"])
    m_traj2 = map(torch.tensor, trajectory_pairs["traj_2"])
    pairs = list(zip(m_traj1, m_traj2))
    return pairs


def sum_over_pairs(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    alpha,
    gamma,
    epsilon,
    device: torch.device,
    batch_size=64,
) -> torch.Tensor:
    """Compute sum of warping distances over given pairs"""
    total = torch.zeros((), device=device)
    finite_cnt = 0
    # batchit = itt.batched(pairs, batch_size)
    for traj_A, traj_B in tqdm(pairs, desc="Computing warping distances", leave=False):
        dist = warping_distance_torch(traj_A[None, :].to(device), traj_B[None, :].to(device), alpha, gamma, epsilon)
        if torch.isfinite(dist):
            total = total + dist
            finite_cnt += 1

    if finite_cnt == 0:
        # Big but finite; avoid overflow and NaNs
        dtype = total.dtype
        finfo = torch.finfo(dtype)
        penalty = torch.tensor(min(1e9, finfo.max / 10), device=device, dtype=dtype)
        return penalty

    return total


def batched_autowarp(
    conn: duckdb.DuckDBPyConnection,
    distparquet_path: str,
    latents: torch.Tensor,
    originals_builder: typing.Callable[[], list[torch.Tensor]],
    device: torch.device,
    pair_batch_size: int = 64,
    max_iters: int = 50,
    convergence_tol: float = 1e-4,
    quantile: float = 0.2,
    lr: float = 1e-2,
):
    """
    Batched Autowarp:
      1) define δ from pairwise latent distances
      2) minimize betaCV over raw α, γ, ε with correct transforms
    """
    print(f"Running Autowarp on device: {device}")

    trajectories = originals_builder()  # list of (Li, D)
    assert len(trajectories) == latents.size(0), "Originals and latents must align."

    quantile = conn.sql(
        f"SELECT approx_quantile(dist, ?::FLOAT) FROM '{distparquet_path}'", params=(quantile,)
    ).fetchone()[0]

    # randomly initialize raw parameters
    raw_alpha = torch.randn((), device=device, requires_grad=True)
    raw_gamma = torch.randn((), device=device, requires_grad=True)
    raw_epsilon = torch.randn((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_alpha, raw_gamma, raw_epsilon], lr=lr)

    print("Starting parameter optimization...")
    prev_beta = None
    current_beta = None

    for step in tqdm(range(max_iters), desc="Autowarp Optimization"):
        # Sample S pairs from close and S from all
        pair_close = sample_close_pairs(conn, distparquet_path, quantile, pair_batch_size)
        pair_all = sample_pairs(conn, distparquet_path, pair_batch_size)
        optimizer.zero_grad()
        num = sum_over_pairs(pair_close, raw_alpha, raw_gamma, raw_epsilon, device=device)
        den = sum_over_pairs(pair_all, raw_alpha, raw_gamma, raw_epsilon, device=device)
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
            print(f"Step {step:3d} | β̂:{current_beta:.4f} | α: {a:.3f} | γ: {g:.3f} | ε: {e:.3f}")

    # Final parameter values
    final_alpha = torch.sigmoid(raw_alpha).item()
    final_gamma = F.softplus(raw_gamma).item()
    final_epsilon = torch.sigmoid(raw_epsilon).item()
    final_beta = current_beta if current_beta is not None else float("nan")

    print("\nFinal Results:")
    print(f"β̂: {final_beta:.4f}")
    print(f"α: {final_alpha:.3f}")
    print(f"γ: {final_gamma:.3f}")
    print(f"ε: {final_epsilon:.3f}")

    return final_beta, final_alpha, final_gamma, final_epsilon
