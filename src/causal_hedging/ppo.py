"""Small Transformer-PPO implementation with bounded categorical hedge decisions."""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .scm import TrainingEnv

HEDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


class Policy(nn.Module):
    def __init__(self, history: int = 104, width: int = 16):
        super().__init__()
        self.input = nn.Linear(3, width)
        position = torch.arange(history).unsqueeze(1)
        scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        encoding = torch.zeros(history, width)
        encoding[:, 0::2], encoding[:, 1::2] = (
            torch.sin(position * scale),
            torch.cos(position * scale),
        )
        self.register_buffer("position", encoding)
        layer = nn.TransformerEncoderLayer(width, 2, 32, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.actor = nn.Linear(width, len(HEDGES))
        self.critic = nn.Linear(width, 1)

    def forward(self, state: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        embedded = self.encoder(self.input(state) + self.position)
        representation = embedded[:, -1]  # last query attends to all 104 available observations
        return Categorical(logits=self.actor(representation)), self.critic(representation).squeeze(
            -1
        )


def gae(rewards, values, dones, bootstrap, gamma=0.99, lam=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    carry, next_value = 0.0, bootstrap
    for i in reversed(range(len(rewards))):
        alive = 1 - dones[i]
        delta = rewards[i] + gamma * next_value * alive - values[i]
        carry = delta + gamma * lam * alive * carry
        advantages[i], next_value = carry, values[i]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def clipped_loss(new_logp, old_logp, advantage, clip=0.2):
    if new_logp.ndim != 1 or new_logp.shape != old_logp.shape or new_logp.shape != advantage.shape:
        raise ValueError("PPO likelihoods and advantages must be matching one-dimensional vectors")
    ratio = (new_logp - old_logp).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()


def train(causal: bool, seed: int, steps: int = 4096, batch_steps: int = 256):
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    env = TrainingEnv(causal, seed)
    policy = Policy(env.config.history)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    state = env.reset()  # persists between rollout batches
    log = []
    for start in range(0, steps, batch_steps):
        states, actions, old_logp, rewards, values, dones = [], [], [], [], [], []
        for _ in range(min(batch_steps, steps - start)):
            with torch.no_grad():
                distribution, value = policy(torch.from_numpy(state[None]))
                action = distribution.sample()
            states.append(state)
            actions.append(action.item())
            old_logp.append(distribution.log_prob(action).item())
            values.append(value.item())
            state, reward, done = env.step(float(HEDGES[action.item()]))
            rewards.append(reward)
            dones.append(done)
            if done:
                state = env.reset()
        with torch.no_grad():
            _, bootstrap = policy(torch.from_numpy(state[None]))
        advantage, returns = gae(rewards, values, dones, bootstrap.item())
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        x = torch.from_numpy(np.stack(states))
        action_tensor = torch.tensor(actions)
        old_tensor = torch.tensor(old_logp)
        adv_tensor, return_tensor = torch.from_numpy(advantage), torch.from_numpy(returns)
        for _ in range(4):
            for ids in torch.randperm(len(states)).split(64):
                distribution, value = policy(x[ids])
                actor_loss = clipped_loss(
                    distribution.log_prob(action_tensor[ids]), old_tensor[ids], adv_tensor[ids]
                )
                value_loss = (value - return_tensor[ids]).square().mean()
                loss = actor_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        log.append({"steps": start + len(states), "mean_utility": float(np.mean(rewards) * 10)})
    policy.eval()
    return policy, log


@torch.no_grad()
def probabilities(policy: Policy, states: np.ndarray) -> np.ndarray:
    policy.eval()
    distribution, _ = policy(torch.from_numpy(states))
    return distribution.probs.numpy()
