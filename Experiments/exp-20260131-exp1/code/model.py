from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softplus(),
        )
        self.trade_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)
        policy = self.actor(hidden)
        trade_logit = self.trade_head(hidden).squeeze(-1)
        value = self.critic(hidden).squeeze(-1)
        return policy, trade_logit, value
