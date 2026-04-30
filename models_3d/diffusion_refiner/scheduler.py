"""
DDPM/DDIM Scheduler for Diffusion Refiner.

Implements the forward and reverse diffusion processes for training and sampling.
Supports both standard DDPM and faster DDIM sampling.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
from .config import DiffusionConfig


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) scheduler.

    Implements:
    - Forward diffusion: q(x_t | x_0) = N(x_t; sqrt(bar_alpha_t) * x_0, (1 - bar_alpha_t) * I)
    - Reverse denoising: p_theta(x_{t-1} | x_t, c) via learned neural network

    Usage:
        scheduler = DDPMScheduler(config)
        scheduler.setup()

        # Training: add noise to clean data
        x_t, noise = scheduler.add_noise(x_0, t, noise)

        # Sampling: denoise step by step
        noise_pred = model(x_t, t, condition)
        x_prev = scheduler.step(noise_pred, t, x_t)
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.beta_schedule = config.beta_schedule
        self.device = None

        self.betas = None
        self.alphas = None
        self.alphas_bar = None
        self.alphas_bar_prev = None
        self.sqrt_alphas_bar = None
        self.sqrt_one_minus_alphas_bar = None
        self.posterior_variance = None
        self.posterior_log_variance = None
        self.posterior_mean_coef1 = None
        self.posterior_mean_coef2 = None

    def setup(self, device: str = "cuda"):
        """Precompute all diffusion schedule parameters."""
        self.device = device

        betas = self.config.get_beta_schedule().to(device)

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1, device=device), alphas_bar[:-1]])

        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.alphas_bar_prev = alphas_bar_prev
        self.sqrt_alphas_bar = sqrt_alphas_bar
        self.sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar

        posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
        posterior_mean_coef2 = (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar)

        self.posterior_variance = posterior_variance
        self.posterior_log_variance = posterior_log_variance
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2

        self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / alphas_bar - 1)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean data at timestep t.

        Args:
            x_0: Clean data [B, C, H, W]
            t: Timesteps [B] (values in [0, num_timesteps-1])
            noise: Optional pre-sampled noise [B, C, H, W]

        Returns:
            x_t: Noisy data at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_bar, t, x_0.shape
        )

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Perform one DDPM denoising step.

        Args:
            model_output: Predicted noise epsilon_theta
            t: Current timestep [B]
            x_t: Noisy data at timestep t
            clip_denoised: Whether to clip predictions to [-1, 1]

        Returns:
            x_{t-1}: Denoised data at previous timestep
        """
        batch_size = x_t.shape[0]

        t = t.long()

        if model_output.shape[1] == x_t.shape[1] * 2:
            model_output, model_var_values = torch.split(model_output, x_t.shape[1], dim=1)
            min_log = self._extract(self.posterior_log_variance, t, x_t.shape)
            max_log = self._extract(torch.log(self.betas), t, x_t.shape)
            model_log_variance = model_var_values * (max_log - min_log) + min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance = self._extract(self.posterior_variance, t, x_t.shape)
            model_log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)

        pred_x0 = self._pred_x0_from_eps(x_t, t, model_output)

        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        noise = torch.randn_like(x_t) if t.sum().item() > 0 else torch.zeros_like(x_t)

        return model_mean + torch.sqrt(model_variance) * noise

    def _pred_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from epsilon (noise)."""
        sqrt_recip_alphas_bar_t = self._extract(self.sqrt_recip_alphas_bar, t, x_t.shape)
        sqrt_recipm1_alphas_bar_t = self._extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)
        return sqrt_recip_alphas_bar_t * x_t - sqrt_recipm1_alphas_bar_t * eps

    def _extract(
        self,
        coefficients: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size
    ) -> torch.Tensor:
        """Extract coefficients at given timesteps and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = coefficients.gather(index=t, dim=0)
        return out.view(batch_size, *([1] * (len(x_shape) - 1)))


class DDIMScheduler(DDPMScheduler):
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler.

    DDIM enables much faster sampling by skipping intermediate steps while
    maintaining the same training objective.

    The trade-off is that DDIM is no longer a probabilistic model but a
    deterministic or semi-deterministic process (controlled by eta).
    """

    def __init__(self, config: DiffusionConfig, eta: float = 0.0):
        super().__init__(config)
        self.eta = eta

    def ddim_step(
        self,
        model_output: torch.Tensor,
        t: int,
        prev_t: int,
        x_t: torch.Tensor,
        clip_denoised: bool = True,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Perform one DDIM denoising step.

        Args:
            model_output: Predicted noise epsilon_theta
            t: Current timestep
            prev_t: Previous timestep (typically t - skip_steps)
            x_t: Noisy data at timestep t
            clip_denoised: Whether to clip predictions to [-1, 1]
            eta: DDIM stochasticity parameter (0=deterministic, 1=full stochastic)

        Returns:
            x_{prev_t}: Denoised data at previous timestep
        """
        if eta is None:
            eta = self.eta

        batch_size = x_t.shape[0]

        t_long = torch.tensor([t] * batch_size, device=x_t.device)
        prev_t_long = torch.tensor([prev_t] * batch_size, device=x_t.device)

        alpha_bar_t = self._extract(self.alphas_bar, t_long, x_t.shape)
        alpha_bar_prev_t = self._extract(self.alphas_bar, prev_t_long, x_t.shape)

        beta_bar_t = 1.0 - alpha_bar_t
        beta_bar_prev_t = 1.0 - alpha_bar_prev_t

        pred_x0 = self._pred_x0_from_eps(x_t, t_long, model_output)

        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        sigma_t = eta * torch.sqrt(
            (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev_t)
        )

        pred_dir_xt_to_x0 = torch.sqrt(1.0 - alpha_bar_prev_t - sigma_t ** 2) * (pred_x0 - x_t) / torch.sqrt(beta_bar_t)

        x_prev = torch.sqrt(alpha_bar_prev_t) * pred_x0 + torch.sqrt(beta_bar_prev_t - sigma_t ** 2) * pred_dir_xt_to_x0

        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    def ddim_sample(
        self,
        model: nn.Module,
        x_T: torch.Tensor,
        condition: torch.Tensor,
        n_steps: Optional[int] = None,
        timesteps: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.

        Args:
            model: Denoising model
            x_T: Initial noise [B, C, H, W]
            condition: Conditioning information [B, C_cond, H, W]
            n_steps: Number of sampling steps (uses config.sampling_steps if None)
            timesteps: Specific timesteps to use [n_steps]
            clip_denoised: Whether to clip predictions to [-1, 1]

        Returns:
            x_0: Denoised sample
        """
        if n_steps is None:
            n_steps = self.config.sampling_steps

        if timesteps is None:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, n_steps, dtype=torch.long, device=x_T.device
            )

        x_t = x_T

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t] * x_T.shape[0], device=x_T.device)

            with torch.no_grad():
                noise_pred = model(x_t, condition, t_tensor)

            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            x_t = self.ddim_step(
                noise_pred,
                t.item(),
                prev_t,
                x_t,
                clip_denoised=clip_denoised,
            )

        return x_t

    def get_timesteps_schedule(
        self,
        n_steps: Optional[int] = None,
        strategy: str = "linear"
    ) -> torch.Tensor:
        """
        Get timesteps schedule for DDIM sampling.

        Args:
            n_steps: Number of steps
            strategy: "linear", "quadratic", or "uniform"

        Returns:
            Timesteps tensor [n_steps]
        """
        if n_steps is None:
            n_steps = self.config.sampling_steps

        if strategy == "linear":
            return torch.linspace(
                self.num_timesteps - 1, 0, n_steps, dtype=torch.long
            )
        elif strategy == "quadratic":
            timesteps = torch.linspace(0, 1, n_steps) ** 2
            return (timesteps * (self.num_timesteps - 1)).long()
        elif strategy == "uniform":
            step_size = self.num_timesteps // n_steps
            return torch.arange(0, self.num_timesteps, step_size, dtype=torch.long)[:n_steps]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for timestep encoding.

    Used to encode diffusion timestep information for the denoiser model.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
