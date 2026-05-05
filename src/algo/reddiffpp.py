import torch
import tqdm
from .base import Algo
import wandb
from utils.scheduler import Scheduler


# -------------------------------------------------------------------------------------------
# Paper: A Variational Perspective on Solving Inverse Problems with Diffusion Models
# Official implementation: https://github.com/NVlabs/RED-diff
# -------------------------------------------------------------------------------------------

class REDDiffPP(Algo):
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
        lambda_schedule_type="reddiffpp",
        lambda_cap=None,
    ):
        super(REDDiffPP, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op

        self.scheduler = Scheduler(num_steps=num_steps, schedule='vp', timestep='vp', scaling='vp')
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        self.base_lambda = base_lambda
        self.sigma_data = sigma_data
        self.lambda_schedule_type = lambda_schedule_type
        self.lambda_cap = None if lambda_cap is None else float(lambda_cap)

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma[:, None, None, None]

    def lambda_fn(self, sigma):
        sigma = torch.as_tensor(sigma)
        if self.lambda_schedule_type == "reddiffpp":
            lam = self.base_lambda / (sigma**2 + self.sigma_data**2)
        elif self.lambda_schedule_type == "constant":
            lam = torch.full_like(sigma, self.base_lambda)
        elif self.lambda_schedule_type == "sqrt":
            lam = self.base_lambda * torch.sqrt(torch.clamp(sigma, min=0))
        elif self.lambda_schedule_type == "linear":
            lam = self.base_lambda * sigma
        else:
            raise NotImplementedError(
                f"Unsupported lambda_schedule_type='{self.lambda_schedule_type}'."
            )

        if self.lambda_cap is not None:
            lam = torch.clamp(lam, max=self.lambda_cap)
        return lam

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)

        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        # 0. random initialization (instead of from pseudo-inverse)
        mu = torch.zeros(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution,
                         device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))

        for step in pbar:
            # 1. forward diffusion
            with torch.no_grad():
                # Get random timesteps
                sigmas = torch.zeros(self.batch_size, device=device)
                scalings = torch.zeros(self.batch_size, device=device)
                
                for i in range(self.batch_size):
                    sigmas[i] = self.scheduler.sigma_steps[step]
                    scalings[i] = self.scheduler.scaling_steps[step]
                mu_expand = mu.repeat(self.batch_size // num_samples, 1, 1, 1)
                epsilon = torch.randn_like(mu_expand)
                xt = scalings[:, None, None, None] * (mu_expand + sigmas[:, None, None, None] * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigmas).detach()

            # 2. regularized optimization
            lam = self.lambda_fn(sigmas) # sigma here equals to 1/SNR
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(mu, observation, return_loss=True)
            gradient = gradient * self.observation_weight + (lam[:, None, None, None] * (pred_epsilon - epsilon)).mean(dim=0, keepdim=True)
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(f'Iteration {step + 1}/{num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)}, step=step)
        return mu
