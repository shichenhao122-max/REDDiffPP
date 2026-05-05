import torch
import tqdm
import wandb

from .reddiffpp import REDDiffPP


class REDDiffPPRSSInit(REDDiffPP):
    """REDDiffPP with configurable MRI initialization strategies."""

    def __init__(
        self,
        net,
        forward_op,
        num_steps=2000,
        batch_size=1,
        observation_weight=1.0,
        base_lambda=0.25,
        base_lr=0.5,
        sigma_data=0.5,
        init_mode="rss",
        init_alpha=1.0,
        rss_scale=1.0,
        phase_mode="mvue",
        fallback_to_zero=False,
        fallback_tol=1.0,
    ):
        super(REDDiffPPRSSInit, self).__init__(
            net=net,
            forward_op=forward_op,
            num_steps=num_steps,
            batch_size=batch_size,
            observation_weight=observation_weight,
            base_lambda=base_lambda,
            base_lr=base_lr,
            sigma_data=sigma_data,
        )
        self.init_mode = init_mode
        self.init_alpha = float(init_alpha)
        self.rss_scale = float(rss_scale)
        self.phase_mode = phase_mode
        self.fallback_to_zero = bool(fallback_to_zero)
        self.fallback_tol = float(fallback_tol)

    def _repeat_init(self, mu, num_samples):
        return mu.repeat(num_samples, 1, 1, 1).detach().clone().requires_grad_(True)

    def _zero_init(self, num_samples, device, dtype):
        mu = torch.zeros(
            1,
            self.net.img_channels,
            self.net.img_resolution,
            self.net.img_resolution,
            device=device,
            dtype=dtype,
        )
        return self._repeat_init(mu, num_samples)

    def _estimated_mvue(self, device):
        estimated_mvue = getattr(self.forward_op, "estimated_mvue", None)
        if estimated_mvue is None:
            raise RuntimeError(
                "MRI initialization expects forward_op.estimated_mvue to be populated "
                "before inference."
            )

        estimated_mvue = torch.as_tensor(estimated_mvue, device=device)
        if not torch.is_complex(estimated_mvue):
            raise RuntimeError("expected forward_op.estimated_mvue to be complex-valued.")
        return estimated_mvue

    def _rss_with_phase_init(self, rss, num_samples, device, dtype):
        phase_source = self._estimated_mvue(device=device)
        phase = phase_source / torch.clamp(phase_source.abs(), min=1e-8)
        init_complex = rss.squeeze(1) * phase.to(dtype=torch.complex64)
        mu = torch.view_as_real(init_complex).permute(0, 3, 1, 2).contiguous()
        mu = mu.to(device=device, dtype=dtype)
        return self._repeat_init(mu, num_samples)

    def _mvue_init(self, num_samples, device, dtype):
        estimated_mvue = self._estimated_mvue(device=device)

        if self.net.img_channels == 1:
            mu = estimated_mvue.abs().unsqueeze(1).to(dtype=dtype)
            return self._repeat_init(mu, num_samples)

        if self.net.img_channels != 2:
            raise NotImplementedError(
                f"REDDiffPPRSSInit only supports 1- or 2-channel priors, got {self.net.img_channels}."
            )

        mu = torch.view_as_real(estimated_mvue.to(dtype=torch.complex64))
        mu = mu.permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)
        return self._repeat_init(mu, num_samples)

    def _rss_init(self, num_samples, device, dtype):
        if not hasattr(self.forward_op, "masked_kspace"):
            raise RuntimeError(
                "REDDiffPPRSSInit expects forward_op.masked_kspace to be populated "
                "by calling forward_op(data) before inference."
            )

        masked_kspace = self.forward_op.masked_kspace
        if not torch.is_complex(masked_kspace):
            raise RuntimeError(
                "REDDiffPPRSSInit expects masked_kspace to be a complex tensor."
            )

        coil_images = self.forward_op.ifft(masked_kspace)
        rss = torch.sqrt(
            torch.clamp(torch.sum(torch.abs(coil_images) ** 2, dim=1, keepdim=True), min=0)
        )
        rss = rss.to(device=device, dtype=dtype) * self.rss_scale

        if self.net.img_channels == 1:
            return self._repeat_init(rss, num_samples)

        if self.net.img_channels != 2:
            raise NotImplementedError(
                f"REDDiffPPRSSInit only supports 1- or 2-channel priors, got {self.net.img_channels}."
            )

        if self.phase_mode == "mvue":
            return self._rss_with_phase_init(rss, num_samples, device, dtype)

        if self.phase_mode != "zero":
            raise NotImplementedError(
                f"Unsupported phase_mode='{self.phase_mode}'. Expected 'zero' or 'mvue'."
            )

        mu = torch.zeros(
            num_samples,
            self.net.img_channels,
            self.net.img_resolution,
            self.net.img_resolution,
            device=device,
            dtype=dtype,
        )
        mu[:, :1] = rss.repeat(num_samples, 1, 1, 1)
        return mu.detach().clone().requires_grad_(True)

    def _build_init(self, observation, num_samples, device, dtype):
        if self.init_mode == "zero":
            mu = self._zero_init(num_samples=num_samples, device=device, dtype=dtype)
        elif self.init_mode == "rss":
            mu = self._rss_init(num_samples=num_samples, device=device, dtype=dtype)
        elif self.init_mode == "mvue":
            mu = self._mvue_init(num_samples=num_samples, device=device, dtype=dtype)
        else:
            raise NotImplementedError(
                f"Unsupported init_mode='{self.init_mode}'. Expected 'zero', 'rss', or 'mvue'."
            )

        if self.init_alpha != 1.0:
            with torch.no_grad():
                mu.mul_(self.init_alpha)

        if not self.fallback_to_zero or self.init_mode == "zero":
            return mu

        zero_mu = self._zero_init(num_samples=num_samples, device=device, dtype=dtype)
        with torch.no_grad():
            init_loss = self.forward_op.loss(mu.detach(), observation).mean()
            zero_loss = self.forward_op.loss(zero_mu.detach(), observation).mean()
            if init_loss > zero_loss * self.fallback_tol:
                return zero_mu
        return mu

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)

        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        mu = self._build_init(
            observation=observation,
            num_samples=num_samples,
            device=device,
            dtype=torch.float32,
        )
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))

        for step in pbar:
            with torch.no_grad():
                sigmas = torch.zeros(self.batch_size, device=device)
                scalings = torch.zeros(self.batch_size, device=device)

                for i in range(self.batch_size):
                    sigmas[i] = self.scheduler.sigma_steps[step]
                    scalings[i] = self.scheduler.scaling_steps[step]
                mu_expand = mu.repeat(self.batch_size // num_samples, 1, 1, 1)
                epsilon = torch.randn_like(mu_expand)
                xt = scalings[:, None, None, None] * (
                    mu_expand + sigmas[:, None, None, None] * epsilon
                )
                pred_epsilon = self.pred_epsilon(self.net, xt, sigmas).detach()

            lam = self.lambda_fn(sigmas)
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(mu, observation, return_loss=True)
            gradient = gradient * self.observation_weight + (
                lam[:, None, None, None] * (pred_epsilon - epsilon)
            ).mean(dim=0, keepdim=True)
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(
                f"Iteration {step + 1}/{num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}"
            )
            if wandb.run is not None:
                wandb.log({"data_fitting_loss": torch.sqrt(loss_scale)}, step=step)
        return mu
