"""Pure PyTorch Go2 Parkour actor shared by Isaac Lab and VLA training."""

from __future__ import annotations

import torch
import torch.nn as nn


GO2_PARKOUR_ACTION_DIM = 12
GO2_PARKOUR_ACTOR_OBSERVATION_DIM = 753
GO2_PARKOUR_HISTORY_LENGTH = 10
GO2_PARKOUR_PRIV_EXPLICIT_DIM = 9
GO2_PARKOUR_PRIV_LATENT_DIM = 29
GO2_PARKOUR_PROPRIO_DIM = 53
GO2_PARKOUR_SCAN_DIM = 132
GO2_PARKOUR_TERRAIN_LATENT_DIM = 32
GO2_PARKOUR_YAW_DIM = 2
GO2_PARKOUR_PERCEPTION_DIM = GO2_PARKOUR_TERRAIN_LATENT_DIM + GO2_PARKOUR_YAW_DIM


class StateHistoryEncoder(nn.Module):
    def __init__(
        self,
        activation_fn: nn.Module,
        input_size: int,
        tsteps: int,
        output_size: int,
        channel_size: int,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size),
            self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=8,
                    stride=4,
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=2 * channel_size,
                    out_channels=channel_size,
                    kernel_size=5,
                    stride=1,
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=channel_size,
                    out_channels=channel_size,
                    kernel_size=5,
                    stride=1,
                ),
                self.activation_fn,
                nn.Flatten(),
            )
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=4,
                    stride=2,
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=2 * channel_size,
                    out_channels=channel_size,
                    kernel_size=2,
                    stride=1,
                ),
                self.activation_fn,
                nn.Flatten(),
            )
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=6,
                    stride=2,
                ),
                self.activation_fn,
                nn.Conv1d(
                    in_channels=2 * channel_size,
                    out_channels=channel_size,
                    kernel_size=4,
                    stride=2,
                ),
                self.activation_fn,
                nn.Flatten(),
            )
        else:
            raise ValueError("tsteps must be 10, 20 or 50")

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size),
            self.activation_fn,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        projection = self.encoder(obs.reshape(batch_size * self.tsteps, -1))
        output = self.conv_layers(
            projection.reshape(batch_size, self.tsteps, -1).permute((0, 2, 1))
        )
        return self.linear_output(output)


class Actor(nn.Module):
    def __init__(
        self,
        num_actions,
        scan_encoder_dims,
        actor_hidden_dims,
        priv_encoder_dims,
        activation,
        tanh_encoder_output=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_prop = num_prop = kwargs.pop("num_prop")
        self.num_scan = num_scan = kwargs.pop("num_scan")
        self.num_hist = num_hist = kwargs.pop("num_hist")
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent = kwargs.pop("num_priv_latent")
        self.num_priv_explicit = num_priv_explicit = kwargs.pop("num_priv_explicit")
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        self.in_features = (
            num_prop
            + num_scan
            + num_priv_latent
            + num_priv_explicit
            + num_prop * num_hist
        )

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for layer_index in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(
                    nn.Linear(priv_encoder_dims[layer_index], priv_encoder_dims[layer_index + 1])
                )
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        state_history_encoder_cfg = kwargs.pop("state_history_encoder")
        state_history_encoder_cfg.pop("class_name")
        self.history_encoder: StateHistoryEncoder = StateHistoryEncoder(
            activation,
            num_prop,
            num_hist,
            priv_encoder_output_dim,
            state_history_encoder_cfg.pop("channel_size"),
        )
        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for layer_index in range(len(scan_encoder_dims) - 1):
                if layer_index == len(scan_encoder_dims) - 2:
                    scan_encoder.append(
                        nn.Linear(scan_encoder_dims[layer_index], scan_encoder_dims[layer_index + 1])
                    )
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(
                        nn.Linear(scan_encoder_dims[layer_index], scan_encoder_dims[layer_index + 1])
                    )
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        actor_layers = []
        actor_layers.append(
            nn.Linear(
                num_prop
                + self.scan_encoder_output_dim
                + num_priv_explicit
                + priv_encoder_output_dim,
                actor_hidden_dims[0],
            )
        )
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1])
                )
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(
        self,
        obs,
        hist_encoding: bool,
        scandots_latent: torch.Tensor | None = None,
    ):
        if self.if_scan_encode:
            obs_scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
            if scandots_latent is None:
                scan_latent = self.scan_encoder(obs_scan)
            else:
                scan_latent = scandots_latent
            obs_prop_scan = torch.cat([obs[:, : self.num_prop], scan_latent], dim=1)
        else:
            obs_prop_scan = obs[:, : self.num_prop + self.num_scan]
        obs_priv_explicit = obs[
            :,
            self.num_prop
            + self.num_scan : self.num_prop
            + self.num_scan
            + self.num_priv_explicit,
        ]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        return self.actor_backbone(backbone_input)

    def infer_priv_latent(self, obs):
        start = self.num_prop + self.num_scan + self.num_priv_explicit
        return self.priv_encoder(obs[:, start : start + self.num_priv_latent])

    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist * self.num_prop :]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
        return self.scan_encoder(scan)


def build_go2_parkour_actor() -> Actor:
    return Actor(
        GO2_PARKOUR_ACTION_DIM,
        scan_encoder_dims=[128, 64, GO2_PARKOUR_TERRAIN_LATENT_DIM],
        actor_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[64, 20],
        activation=nn.ELU(),
        num_prop=GO2_PARKOUR_PROPRIO_DIM,
        num_scan=GO2_PARKOUR_SCAN_DIM,
        num_hist=GO2_PARKOUR_HISTORY_LENGTH,
        num_priv_latent=GO2_PARKOUR_PRIV_LATENT_DIM,
        num_priv_explicit=GO2_PARKOUR_PRIV_EXPLICIT_DIM,
        state_history_encoder={"class_name": "StateHistoryEncoder", "channel_size": 10},
    )


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def apply_parkour_mts(
    actor_observation: torch.Tensor,
    predicted_yaw: torch.Tensor,
    yaw_scale: float = 1.5,
    mts_threshold: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    observation = actor_observation.clone()
    current_yaw_error = _wrap_to_pi(
        yaw_scale * predicted_yaw[..., 0] - observation[..., 6]
    )
    use_prediction = current_yaw_error.abs() < mts_threshold
    observation[..., 6:8] = torch.where(
        use_prediction[..., None],
        yaw_scale * predicted_yaw,
        observation[..., 6:8],
    )
    return observation, use_prediction


__all__ = [
    "Actor",
    "StateHistoryEncoder",
    "GO2_PARKOUR_ACTION_DIM",
    "GO2_PARKOUR_ACTOR_OBSERVATION_DIM",
    "GO2_PARKOUR_HISTORY_LENGTH",
    "GO2_PARKOUR_PERCEPTION_DIM",
    "GO2_PARKOUR_PRIV_EXPLICIT_DIM",
    "GO2_PARKOUR_PRIV_LATENT_DIM",
    "GO2_PARKOUR_PROPRIO_DIM",
    "GO2_PARKOUR_SCAN_DIM",
    "GO2_PARKOUR_TERRAIN_LATENT_DIM",
    "GO2_PARKOUR_YAW_DIM",
    "apply_parkour_mts",
    "build_go2_parkour_actor",
]
