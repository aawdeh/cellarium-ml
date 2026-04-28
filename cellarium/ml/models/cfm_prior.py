# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Conditional Flow Matching (CFM) as a learned prior over the scVI latent space z.

CFM learns p(z) by training a vector field MLP that transports a simple source
distribution (Normal(0,I)) to the data distribution of encoder z values.

Key difference from DDPM (diffusion_prior.py):
  - DDPM: T=1000 discrete noising steps, reverse via iterative denoising
  - CFM:  continuous straight-line paths from noise to data, reverse via ODE integration
           at inference you can use as few as 10-50 ODE steps vs 1000 DDPM steps

Training objective (flow matching loss):
  Given z0 ~ Normal(0,I) and z1 ~ q(z|x) (encoder output),
  define a straight-line interpolant at time t in [0,1]:
      z_t = (1 - t) * z0  +  t * z1
  The target velocity is the time derivative:
      u_t = z1 - z0  (constant along the path — the CFM simplification)
  Train MLP v_theta(z_t, t) to predict u_t.
  Loss = MSE(v_theta(z_t, t), u_t)

References:
  Lipman et al. "Flow Matching for Generative Modeling" (2022)
  Albergo & Vanden-Eijnden "Building Normalizing Flows with Stochastic Interpolants" (2022)
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous time t in [0, 1].
    Same principle as the timestep embedding in diffusion_prior.py but
    t is a continuous float here, not a discrete integer.

    Args:
        dim: output embedding dimension (should be even)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (n,) float times in [0, 1], one per cell
        Returns:
            (n, dim) float embedding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]  # (n, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (n, dim)


class CFMPrior(nn.Module):
    """Conditional Flow Matching prior over the scVI latent space z.
    Replaces the fixed Normal(0, I) prior — same role as LatentDiffusionPrior
    but uses flow matching instead of DDPM.

    Contains:
      - SinusoidalTimeEmbedding: converts continuous t to a float vector
      - A velocity field MLP: v_theta(z_t, t) -> velocity (learned params)

    The MLP is small because z is only n_latent=10-30 dimensional.

    Args:
        n_latent:    dimensionality of z (must match scVI n_latent)
        t_emb_dim:   sinusoidal embedding size for t
        hidden_dims: MLP hidden layer widths
        n_ode_steps: number of Euler steps used in sample() and log_prob_approx()
                     More steps = more accurate, slower.
                     CFM typically needs far fewer steps than DDPM (10-50 vs 1000).
        sigma_min:   minimum std of the source Gaussian at t=1 (OT-CFM style);
                     0.0 uses exact straight-line paths (standard CFM)
    """

    def __init__(
        self,
        n_latent: int,
        t_emb_dim: int = 128,
        hidden_dims: list[int] = [512, 512],
        n_ode_steps: int = 50,
        sigma_min: float = 0.0,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_ode_steps = n_ode_steps
        self.sigma_min = sigma_min

        self.t_embed = SinusoidalTimeEmbedding(t_emb_dim)

        # velocity field MLP: [z_t | t_emb] -> velocity, same shape as z
        in_dim = n_latent + t_emb_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, n_latent))
        self.velocity_field = nn.Sequential(*layers)

    def _predict_velocity(self, z_t_nk: torch.Tensor, t_n: torch.Tensor) -> torch.Tensor:
        """Run the velocity field MLP.

        Args:
            z_t_nk: (n, K) interpolated latent at time t
            t_n:    (n,)   continuous times in [0, 1]
        Returns:
            (n, K) predicted velocity
        """
        t_emb = self.t_embed(t_n)                                # (n, t_emb_dim)
        h = torch.cat([z_t_nk, t_emb], dim=-1)                  # (n, K + t_emb_dim)
        return self.velocity_field(h)                            # (n, K)

    def _interpolate(
        self,
        z0_nk: torch.Tensor,
        z1_nk: torch.Tensor,
        t_n: torch.Tensor,
    ) -> torch.Tensor:
        """Straight-line interpolant with optional sigma_min perturbation.

        Standard CFM (sigma_min=0):
            z_t = (1 - t) * z0  +  t * z1

        OT-CFM (sigma_min > 0) adds a small Gaussian perturbation to keep
        paths from collapsing exactly at t=1:
            z_t = (1 - (1 - sigma_min) * t) * z0  +  t * z1

        Args:
            z0_nk: (n, K) source samples ~ Normal(0, I)
            z1_nk: (n, K) target samples ~ q(z|x) from encoder
            t_n:   (n,)   times in [0, 1]
        Returns:
            (n, K) interpolated latent at time t
        """
        t = t_n[:, None]  # (n, 1) for broadcasting
        if self.sigma_min == 0.0:
            return (1.0 - t) * z0_nk + t * z1_nk
        else:
            return (1.0 - (1.0 - self.sigma_min) * t) * z0_nk + t * z1_nk

    def _target_velocity(
        self,
        z0_nk: torch.Tensor,
        z1_nk: torch.Tensor,
    ) -> torch.Tensor:
        """The target velocity u_t — what the MLP should predict.

        Standard CFM: u_t = z1 - z0  (constant along each path, t-independent)
        OT-CFM:       u_t = z1 - (1 - sigma_min) * z0

        Args:
            z0_nk: (n, K) source samples
            z1_nk: (n, K) target samples
        Returns:
            (n, K) target velocity
        """
        if self.sigma_min == 0.0:
            return z1_nk - z0_nk
        else:
            return z1_nk - (1.0 - self.sigma_min) * z0_nk

    def flow_matching_loss(self, z1_nk: torch.Tensor) -> torch.Tensor:
        """Flow matching training loss — trains the velocity field MLP.

        Called in scVI forward() with z1 detached from the encoder graph,
        so the encoder does not receive gradients through this loss.

        Steps:
          1. Sample source noise z0 ~ Normal(0, I), same shape as z1
          2. Sample random time t ~ Uniform(0, 1) per cell
          3. Interpolate z_t = (1-t)*z0 + t*z1
          4. Predict velocity v_theta(z_t, t)
          5. Return MSE(predicted_velocity, target_velocity)

        Args:
            z1_nk: (n, K) encoder z samples, DETACHED from encoder
        Returns:
            scalar MSE loss
        """
        n = z1_nk.shape[0]
        z0_nk = torch.randn_like(z1_nk)                                  # source noise
        t_n = torch.rand(n, device=z1_nk.device)                         # (n,) uniform in [0,1]

        z_t_nk = self._interpolate(z0_nk, z1_nk, t_n)
        u_t_nk = self._target_velocity(z0_nk, z1_nk)

        v_pred_nk = self._predict_velocity(z_t_nk, t_n)
        return nn.functional.mse_loss(v_pred_nk, u_t_nk)                 # scalar

    def log_prob_approx(self, z1_nk: torch.Tensor) -> torch.Tensor:
        """Approximate log p(z1) per cell for the KL term in the scVI ELBO.

        Uses the instantaneous change-of-variables formula from continuous
        normalizing flows:
            log p(z1) = log p(z0) - integral_0^1 div(v_theta(z_t, t)) dt

        where div is the divergence of the velocity field (trace of Jacobian),
        approximated via Hutchinson's trace estimator:
            div(v) ≈ epsilon^T (dv/dz) epsilon,  epsilon ~ Normal(0, I)

        This is exact in expectation and requires only one backward pass per
        ODE step, making it tractable even for large n_latent.

        Args:
            z1_nk: (n, K) latent samples from the encoder (with gradient)
        Returns:
            (n,) approximate log p(z1) per cell
        """
        n = z1_nk.shape[0]

        # run the ODE backwards: from t=1 (data) to t=0 (source noise)
        # accumulate log-det along the way
        z = z1_nk.clone()
        log_det_accum = torch.zeros(n, device=z1_nk.device)
        dt = 1.0 / self.n_ode_steps

        for step in reversed(range(self.n_ode_steps)):
            t_val = (step + 1) / self.n_ode_steps                        # t in (0, 1]
            t_n = torch.full((n,), t_val, device=z.device)

            # Hutchinson divergence estimate: eps^T J eps
            eps = torch.randn_like(z)
            with torch.enable_grad():
                z_in = z.detach().requires_grad_(True)
                v = self._predict_velocity(z_in, t_n)
                # vector-Jacobian product: grad of (eps^T v) w.r.t. z_in
                vjp = torch.autograd.grad(
                    (v * eps).sum(), z_in, create_graph=False
                )[0]
            div_v = (vjp * eps).sum(dim=-1)                              # (n,) Hutchinson estimate

            # accumulate log-det (negative because we go t=1 -> t=0)
            log_det_accum -= div_v * dt

            # Euler step backwards
            with torch.no_grad():
                v_step = self._predict_velocity(z, t_n)
            z = z - v_step * dt                                          # (n, K)

        # z is now approximately z0 at t=0 — evaluate log p(z0) = Normal(0,I)
        log_p_z0 = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)  # (n,)

        # log p(z1) = log p(z0) + log |det J|
        return log_p_z0 + log_det_accum                                  # (n,)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate n new latent vectors z ~ p(z) via ODE integration.
        Used at generation time — not called during training.

        Integrates the learned velocity field from t=0 (source noise)
        to t=1 (data distribution) using simple Euler steps.

        Args:
            n:      number of cells to generate
            device: torch device
        Returns:
            (n, K) synthetic latent vectors
        """
        z = torch.randn(n, self.n_latent, device=device)  # start: Normal(0, I)
        dt = 1.0 / self.n_ode_steps

        for step in range(self.n_ode_steps):
            t_val = step / self.n_ode_steps                              # t in [0, 1)
            t_n = torch.full((n,), t_val, device=device)
            v = self._predict_velocity(z, t_n)
            z = z + v * dt                                               # Euler step forward

        return z  # (n, K)