"""Parkour command actor used by the latency robustness PPO run."""

from __future__ import annotations

import torch

from parkour_isaaclab.actor import Actor, GO2_PARKOUR_YAW_SCALE


class CommandActor(Actor):
    """Predict one latent+yaw command from the teacher feature modules."""

    def __init__(self, num_actions, *args, action_horizon: int = 1, command_dim: int = 34, **kwargs):
        self.action_horizon = action_horizon
        self.command_dim = command_dim
        super().__init__(action_horizon * command_dim, *args, **kwargs)
        last = self.actor_backbone[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def forward(self, obs, hist_encoding: bool, scandots_latent=None):
        if self.if_scan_encode:
            obs_scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
            scan_latent = self.scan_encoder(obs_scan) if scandots_latent is None else scandots_latent
            obs_prop_scan = torch.cat([obs[:, : self.num_prop], scan_latent], dim=1)
        else:
            scan_latent = obs[:, self.num_prop : self.num_prop + self.num_scan]
            obs_prop_scan = obs[:, : self.num_prop + self.num_scan]
        obs_priv_explicit = obs[
            :, self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.num_priv_explicit
        ]
        latent = self.infer_hist_latent(obs) if hist_encoding else self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        residual = self.actor_backbone(backbone_input).reshape(
            obs.shape[0], self.action_horizon, self.command_dim
        )
        current = torch.cat(
            (scan_latent, obs[:, 6:8] / GO2_PARKOUR_YAW_SCALE), dim=-1
        )
        return (current.unsqueeze(1) + residual).reshape(obs.shape[0], -1)
