import torch
import tqdm
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_latent_representations(trajectories, encoder, max_len):
    encoder.eval()
    latents = []
    with torch.no_grad():
        for traj in trajectories:
            input_seq = torch.tensor(traj[:max_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
            _, hidden = encoder(input_seq)
            h_n, _ = hidden
            vec = h_n.squeeze()   # No .cpu().numpy()
            latents.append(vec)
    return torch.stack(latents)


def soft_dtw_matrix(x, y, alpha, gamma, epsilon):
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)

    if x.size(0) < 2 or y.size(0) < 2:
        print("Skipping short sequences:", x.size(0), y.size(0))
        return alpha * 0.0 + gamma * 0.0 + epsilon * 0.0

    L1, L2 = x.size(0), y.size(0)
    D = torch.cdist(x.unsqueeze(1), y.unsqueeze(1), p=2)
    D = alpha * D + gamma
    if torch.isnan(D).any() or torch.isinf(D).any():
        print("D contains nan/inf at start")
        return alpha * 0.0 + gamma * 0.0 + epsilon * 0.0

    # Initialize R first, then set [0,0]
    R = torch.zeros((L1 + 1, L2 + 1), device=x.device, dtype=torch.float64) + float("inf")
    R[0, 0] = 0.0
    R[1:, 0] = float('inf')
    R[0, 1:] = float('inf')

    epsilon = torch.clamp(epsilon, min=1e-3)

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            rmin = torch.stack([
                -R[i - 1, j],
                -R[i, j - 1],
                -R[i - 1, j - 1]
            ])
            finite_mask = torch.isfinite(rmin)
            if not finite_mask.any():
                R[i, j] = 1e6  # No valid path, assign large value
                continue
            softmin = -epsilon * torch.logsumexp(rmin[finite_mask] / -epsilon, dim=0)
            if torch.isnan(softmin) or torch.isinf(softmin):
                print(f"NaN/Inf detected at i={i}, j={j}, rmin={rmin}, epsilon={epsilon}")
                return alpha * 0.0 + gamma * 0.0 + epsilon * 0.0 
            R[i, j] = D[i - 1, j - 1] + softmin
            if torch.isnan(R[i, j]) or torch.isinf(R[i, j]):
                print(f"NaN/Inf in R at {i},{j}")
                return alpha * 0.0 + gamma * 0.0 + epsilon * 0.0 

    return R[L1, L2]


def compute_distance_matrix(latents):
    # Ensure latents is on CPU and NumPy for scipy
    latents_np = latents.detach().cpu().numpy()
    return squareform(pdist(latents_np, metric='euclidean'))


def sample_pairs(dist_matrix, threshold, num_pairs):
    N = dist_matrix.shape[0]
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    close_pairs = [pair for pair in all_pairs if dist_matrix[pair[0], pair[1]] < threshold]

    Pc = random.sample(close_pairs, min(len(close_pairs), num_pairs))
    Pall = random.sample(all_pairs, min(len(all_pairs), num_pairs))
    return Pc, Pall


def autowarp_distance(latent_i, latent_j, alpha, gamma, epsilon):
    return soft_dtw_matrix(latent_i, latent_j, alpha, gamma, epsilon)


def align_warping(latents, alpha, gamma, epsilon, pairs):
    total = torch.tensor(0.0, device=DEVICE)
    count = 0
    for i, j in pairs:
        xi, xj = latents[i], latents[j]
        if xi.shape[0] < 2 or xj.shape[0] < 2:
            continue
        d = autowarp_distance(xi, xj, alpha, gamma, epsilon)
        if torch.isnan(d) or torch.isinf(d):
            continue
        total += d.squeeze() if d.dim() > 0 else d
        count += 1
    return total  # return sum not mean


def batched_autowarp(
    trajectories,
    encoder,
    max_len,
    percentile=20,
    lr=0.01,
    batch_size=32,
    max_iters=100,
    convergence_tol=1e-4
):
    # latent representations
    latents = compute_latent_representations(trajectories, encoder, max_len)
    dist_matrix = compute_distance_matrix(latents)

    # distance threshold δ
    distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    threshold = np.percentile(distances, percentile)
    # initialize parameters α, γ, ε randomly between 0 and 1
    alpha = torch.rand(1, requires_grad=True, device=DEVICE)
    gamma = torch.rand(1, requires_grad=True, device=DEVICE)
    epsilon = torch.rand(1, requires_grad=True, device=DEVICE)

    optimizer = torch.optim.SGD([alpha, gamma, epsilon], lr=lr)

    prev_beta = None
    beta_val = 0.0  # initialize beta_val to ensure it is always defined

    for step in tqdm.tqdm(range(max_iters), desc="Autowarp optimization"):
        # sample trajectory pairs
        Pc, Pall = sample_pairs(dist_matrix, threshold, batch_size)

        optimizer.zero_grad()

        sum_close = align_warping(latents, alpha, gamma, epsilon, Pc)
        sum_all = align_warping(latents, alpha, gamma, epsilon, Pall)
        beta_cv = sum_close / (sum_all + 1e-8)

        # gradient descent
        beta_cv.backward()
        optimizer.step()

        beta_val = beta_cv.item()

        # convergence check
        if prev_beta is not None and abs(beta_val - prev_beta) < convergence_tol:
            print(f"Converged at step {step} with β̂ ≈ {beta_val:.4f}")
            break
        prev_beta = beta_val

        if step % 10 == 0:
            tqdm.tqdm.write(
                f"Step {step} — β̂: {beta_val:.4f}, "
                f"α: {alpha.item():.3f}, γ: {gamma.item():.3f}, ε: {epsilon.item():.3f}"
            )

    print("Finished optimization.")
    print(f"β̂: {beta_val:.4f}, "
                f"α: {alpha.item():.3f}, γ: {gamma.item():.3f}, ε: {epsilon.item():.3f}")
    return beta_val, alpha.item(), gamma.item(), epsilon.item()