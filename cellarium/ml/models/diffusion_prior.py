# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion model as a learned prior over the scVI latent space z."""
 
import math
 
import torch
import torch.nn as nn

# Step 1 — SinusoidalTimestepEmbedding: turn a number into a vector
class SinusoidalTimestepEmbedding(nn.Module):
    """
    Converts integer timestep t into a continuous vector via sinusoidal embedding.
    No learned parameters — pure deterministic math, same idea as transformer positional encodings.
 
    Args:
        dim: output embedding dimension (should be even)
    """
 
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
 
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (n,) integer timesteps, one per cell in the batch
        Returns:
            (n, dim) float embedding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]  # (n, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (n, dim)
 
 
# Step 2 — GaussianDiffusion: corrupt a clean z into a noisy z_t
class GaussianDiffusion(nn.Module):
    """
    Linear (DDPM) noise schedule and forward process math.
    No learned parameters — registers alpha/beta tensors as buffers so they
    move to GPU with the model but are not trained.
 
    Args:
        T: total diffusion timesteps
        beta_start: noise variance at t=1 (small — barely noisy)
        beta_end: noise variance at t=T (large — near pure noise)
    """
 
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T
 
        betas = torch.linspace(beta_start, beta_end, T)       # (T,)  noise added at each step
        alphas = 1.0 - betas                                   # (T,)  signal kept at each step
        alpha_bar = torch.cumprod(alphas, dim=0)               # (T,)  cumulative signal retention
 
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
 
    def q_sample(
        self,
        z0_nk: torch.Tensor,
        t_n: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: corrupt z0 to z_t in one step.
            z_t = sqrt(ᾱ_t) * z0  +  sqrt(1 - ᾱ_t) * ε
 
        Args:
            z0_nk: (n, K) clean latent samples from the encoder
            t_n:   (n,)   integer timestep per cell
            noise: (n, K) optional pre-sampled noise; sampled fresh if None
        Returns:
            z_t_nk: (n, K) noisy latent at timestep t
            noise:  (n, K) the noise that was added (target for denoiser)
        """
        if noise is None:
            noise = torch.randn_like(z0_nk)
        ab = self.alpha_bar[t_n].float()[:, None]              # (n, 1) broadcast over K
        z_t_nk = ab.sqrt() * z0_nk + (1.0 - ab).sqrt() * noise
        return z_t_nk, noise
 

class LatentDiffusionPrior(nn.Module):
    """
    Diffusion model that learns p(z) — the prior over the scVI latent space.
    Replaces the fixed Normal(0, I) prior.
 
    Contains:
      - GaussianDiffusion: the noise schedule (no learned params)
      - A denoiser MLP: ε_θ(z_t, t) → predicted noise (learned params)
 
    The MLP is small because z is only n_latent=10–30 dimensional.
 
    Args:
        n_latent:    dimensionality of z (must match scVI n_latent)
        T:           diffusion timesteps
        t_emb_dim:   sinusoidal embedding size for t
        hidden_dims: list of hidden layer widths for the denoiser MLP
        beta_start:  noise schedule start
        beta_end:    noise schedule end
        n_log_prob_timesteps: number of Monte Carlo timesteps used in
                     log_prob_approx (more = lower variance estimate, slower)
    """
 
    def __init__(
        self,
        n_latent: int,
        T: int = 1000,
        t_emb_dim: int = 128,
        hidden_dims: list[int] = [512, 512],
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        n_log_prob_timesteps: int = 10,
    ):
        super().__init__()
        self.T = T
        self.n_latent = n_latent
        self.n_log_prob_timesteps = n_log_prob_timesteps
 
        self.diffusion = GaussianDiffusion(T=T, beta_start=beta_start, beta_end=beta_end)
        self.t_embed = SinusoidalTimestepEmbedding(t_emb_dim)
 
        # denoiser MLP: [z_t | t_emb] -> predicted noise, same shape as z
        in_dim = n_latent + t_emb_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, n_latent))
        self.denoiser = nn.Sequential(*layers)
 
    def _predict_noise(self, z_t_nk: torch.Tensor, t_n: torch.Tensor) -> torch.Tensor:
        """
        Run the denoiser: predict the noise ε given noisy z_t and timestep t.
 
        Args:
            z_t_nk: (n, K) noisy latent
            t_n:    (n,)   integer timesteps
        Returns:
            (n, K) predicted noise
        """
        t_emb = self.t_embed(t_n)                                   # (n, t_emb_dim)
        h = torch.cat([z_t_nk, t_emb], dim=-1)                     # (n, K + t_emb_dim)
        return self.denoiser(h)                                      # (n, K)
 
    def diffusion_loss(self, z0_nk: torch.Tensor) -> torch.Tensor:
        """
        DDPM denoising loss — trains the denoiser parameters.
 
        Called in scVI forward() with z0 detached from the encoder graph,
        so the encoder does not receive gradients through this loss.
 
        Steps:
          1. Sample a random timestep t for each cell
          2. Corrupt z0 → z_t via the forward process
          3. Ask denoiser to predict the noise
          4. Return MSE(predicted_noise, actual_noise)
 
        Args:
            z0_nk: (n, K) clean latent samples, DETACHED from encoder
        Returns:
            scalar MSE loss
        """
        n = z0_nk.shape[0]
        t_n = torch.randint(0, self.T, (n,), device=z0_nk.device)  # (n,) random timesteps
        z_t_nk, noise = self.diffusion.q_sample(z0_nk, t_n)        # corrupt z0
        noise_pred = self._predict_noise(z_t_nk, t_n)              # predict noise
        return nn.functional.mse_loss(noise_pred, noise)            # scalar
 
    def log_prob_approx(self, z0_nk: torch.Tensor) -> torch.Tensor:
        """
        Approximate log p(z0) per cell, used to compute the KL term in the scVI ELBO.
 
        The diffusion model does not give log p(z) in closed form. Instead, we
        approximate it via the diffusion VLB: the higher the denoiser's MSE on z0,
        the less z0 looks like the learned prior, so the lower log p(z0) is.
 
        Concretely: average over n_log_prob_timesteps random t values,
        and return -MSE as a proxy for log p(z0 | t). This is proportional to
        the true VLB term and sufficient for optimisation.
 
        Args:
            z0_nk: (n, K) latent samples from the encoder (with grad)
        Returns:
            (n,) approximate log p(z0) per cell — higher = more likely under prior
        """
        n = z0_nk.shape[0]
        log_p_n = torch.zeros(n, device=z0_nk.device)
 
        for _ in range(self.n_log_prob_timesteps):
            t_n = torch.randint(0, self.T, (n,), device=z0_nk.device)
            z_t_nk, noise = self.diffusion.q_sample(z0_nk, t_n)
            noise_pred = self._predict_noise(z_t_nk, t_n)
            # per-cell sum of squared errors (sum over K dims, not mean)
            mse_per_cell = ((noise_pred - noise) ** 2).sum(dim=-1)  # (n,)
            log_p_n -= mse_per_cell
 
        return log_p_n / self.n_log_prob_timesteps                   # (n,)
 
    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Generate n new latent vectors z ~ p(z) via DDPM reverse process.
        Used at generation time — not called during training.
 
        Start from pure Gaussian noise, iteratively denoise for T steps.
        The resulting z vectors can be passed to DecoderSCVI to get gene counts.
 
        Args:
            n:      number of cells to generate
            device: torch device
        Returns:
            (n, K) synthetic latent vectors
        """
        z = torch.randn(n, self.n_latent, device=device)             # start from noise
 
        for i in reversed(range(self.T)):
            t_n = torch.full((n,), i, device=device, dtype=torch.long)
            noise_pred = self._predict_noise(z, t_n)
 
            alpha_t = self.diffusion.alphas[i]
            alpha_bar_t = self.diffusion.alpha_bar[i]
            beta_t = self.diffusion.betas[i]
 
            # DDPM reverse step (Eq. 11 in Ho et al. 2020)
            z = (1.0 / alpha_t.sqrt()) * (
                z - (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt() * noise_pred
            )
            if i > 0:
                z += beta_t.sqrt() * torch.randn_like(z)             # add noise except last step
 
        return z