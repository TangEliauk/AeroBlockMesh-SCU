"""Lightweight Dual-Branch Continuous SAC networks and replay buffer for CH-SAC v2."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ch_env import ACTION_DIM, GEOM_DIM, OBS_DIM

STATE_DIM = OBS_DIM - GEOM_DIM


class DualBranchEncoder(nn.Module):
    """Separate geometry and mesh-state encoders fused via LayerNorm + SiLU."""

    def __init__(self, geom_dim=GEOM_DIM, state_dim=STATE_DIM, hidden=128):
        super().__init__()
        self.geom_dim = geom_dim
        self.geom_net = nn.Sequential(
            nn.Linear(geom_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def forward(self, obs):
        g_emb = self.geom_net(obs[:, : self.geom_dim])
        s_emb = self.state_net(obs[:, self.geom_dim :])
        return self.fusion(torch.cat([g_emb, s_emb], dim=-1))


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy with exact log-prob Jacobian correction."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, hidden=128, action_dim=ACTION_DIM):
        super().__init__()
        self.encoder = DualBranchEncoder(hidden=hidden)
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        h = self.encoder(obs)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, deterministic=False):
        mu, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        act = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - act.pow(2) + 1e-6)
        return act, log_prob.sum(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    def __init__(self, geom_dim=GEOM_DIM, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.geom_dim = geom_dim
        self.geom_net = nn.Sequential(
            nn.Linear(geom_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + action_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        g_emb = self.geom_net(obs[:, : self.geom_dim])
        s_emb = self.state_net(obs[:, self.geom_dim :])
        return self.head(torch.cat([g_emb, s_emb, action], dim=-1))


class TwinQCritic(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.q1 = QNetwork(hidden=hidden)
        self.q2 = QNetwork(hidden=hidden)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)


class ReplayBuffer:
    def __init__(self, capacity=50000, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.nxt = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, act, rew, nxt, done):
        n = len(obs)
        for i in range(n):
            idx = self.ptr
            self.obs[idx] = obs[i]
            self.act[idx] = act[i]
            self.rew[idx, 0] = rew[i]
            self.nxt[idx] = nxt[i]
            self.done[idx, 0] = float(done[i])
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng, device):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.act[idx], device=device),
            torch.as_tensor(self.rew[idx], device=device),
            torch.as_tensor(self.nxt[idx], device=device),
            torch.as_tensor(self.done[idx], device=device),
        )


class CHSACAgent:
    """Continuous SAC trainer with automatic entropy tuning and target Polyak averaging."""

    def __init__(self, device, lr=3e-4, gamma=0.95, tau=0.005, total_updates=24000):
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_entropy = -float(ACTION_DIM)

        self.actor = SquashedGaussianActor().to(self.device)
        self.critic = TwinQCritic().to(self.device)
        self.critic_target = TwinQCritic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(math.log(0.15), dtype=torch.float32, device=self.device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt, T_max=max(1, total_updates), eta_min=3e-5
        )
        self.critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_opt, T_max=max(1, total_updates), eta_min=3e-5
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        act, _ = self.actor.sample(t, deterministic=deterministic)
        return act.cpu().numpy()

    def update(self, obs, act, rew, nxt, done):
        alpha_val = self.alpha.detach()
        with torch.no_grad():
            nxt_act, nxt_logp = self.actor.sample(nxt, deterministic=False)
            tq1, tq2 = self.critic_target(nxt, nxt_act)
            tq = torch.min(tq1, tq2) - alpha_val * nxt_logp
            target_q = rew + (1.0 - done) * self.gamma * tq

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        pi_act, pi_logp = self.actor.sample(obs, deterministic=False)
        q1_pi, q2_pi = self.critic(obs, pi_act)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha_val * pi_logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (pi_logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        self.actor_sched.step()
        self.critic_sched.step()
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q_pi.mean().item()),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": float(self.log_alpha.item()),
        }

    def load_actor(self, ckpt_path):
        data = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(data["actor"])
