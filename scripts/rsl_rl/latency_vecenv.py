"""RSL-RL vector adapter for Parkour latent command training."""

from __future__ import annotations

import torch

from training.common.command_latency import CommandLatencyBatch

from .vecenv_wrapper import ParkourRslRlVecEnvWrapper
from parkour_isaaclab.actor import GO2_PARKOUR_YAW_SCALE


class ParkourLatencyRslRlVecEnvWrapper(ParkourRslRlVecEnvWrapper):
    """Consume delayed latent+yaw commands while keeping the native decoder fixed."""

    def __init__(
        self,
        env,
        decoder,
        latency_config: dict,
        *,
        action_horizon: int = 40,
        control_repeat: int = 5,
        gamma: float = 0.99,
        clip_actions: float | None = None,
    ):
        super().__init__(env, clip_actions=None)
        self.clip_actions = clip_actions
        self.decoder = decoder.eval().requires_grad_(False)
        self.control_repeat = control_repeat
        self.gamma = gamma
        self.action_horizon = action_horizon
        self.command_dim = 34
        self.num_actions = self.command_dim * action_horizon
        self._scheduler = CommandLatencyBatch(
            latency_config,
            num_envs=self.num_envs,
            device=self.device,
            noop_command=torch.zeros(self.command_dim, device=self.device),
        )
        self._scheduler.reset()
        self._raw_obs, _ = self.get_observations()

    def step(self, commands: torch.Tensor):
        if self.clip_actions is not None:
            commands = torch.clamp(commands, -self.clip_actions, self.clip_actions)
        commands = commands.reshape(self.num_envs, self.action_horizon, self.command_dim)
        reward = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        truncation = torch.zeros_like(dones)
        executed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        terminal_obs = None
        extras = {}
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        active_ids = torch.nonzero(active, as_tuple=False).flatten()
        admission = self._scheduler.submit(commands, env_ids=active_ids.tolist())
        for raw_frame in range(self.control_repeat):
            active_ids = torch.nonzero(active, as_tuple=False).flatten()
            if not active_ids.numel():
                break
            applied = self._scheduler.actions(env_ids=active_ids.tolist())
            latent = applied[:, :32]
            yaw = applied[:, 32:] * GO2_PARKOUR_YAW_SCALE
            actor_obs = self._raw_obs.clone()
            actor_obs[:, 6:8] = yaw
            with torch.inference_mode():
                motor_action = self.decoder(
                    actor_obs,
                    hist_encoding=True,
                    scandots_latent=latent,
                )
            obs_dict, frame_reward, terminated, truncated, extras = self.unwrapped.step_no_reset(
                motor_action, active_mask=active
            )
            self._raw_obs = obs_dict["policy"]
            frame_done = terminated | truncated
            frame_discount = self.gamma**raw_frame
            reward[active] += frame_discount * frame_reward[active]
            executed_steps[active] += 1
            dones[active] |= frame_done[active].to(torch.long)
            truncation[active] |= truncated[active].to(torch.long)
            newly_done = active & frame_done
            if newly_done.any():
                if terminal_obs is None:
                    terminal_obs = {key: value.clone() for key, value in obs_dict.items()}
                else:
                    for key, value in obs_dict.items():
                        terminal_obs[key][newly_done] = value[newly_done]
            active &= ~frame_done
            self._scheduler.advance(env_ids=active_ids.tolist())
        if terminal_obs is not None:
            extras["terminal_observations"] = terminal_obs
        extras["latency_executed_steps"] = executed_steps
        extras["latency_discount"] = torch.pow(
            torch.full_like(executed_steps, self.gamma, dtype=torch.float32),
            executed_steps,
        )
        extras["latency_admission"] = admission
        done_ids = torch.nonzero(dones.bool(), as_tuple=False).flatten().tolist()
        if done_ids:
            reset_obs, _ = self.unwrapped.reset(env_ids=torch.as_tensor(done_ids, device=self.device))
            self._raw_obs = reset_obs["policy"]
            obs_dict = reset_obs
            self._scheduler.reset(done_ids)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncation.bool()
        return self._raw_obs, reward, dones, extras

    def reset(self):
        obs, extras = super().reset()
        self._scheduler.reset()
        self._raw_obs = obs
        return obs, extras

    def latency_state_dict(self):
        return self._scheduler.state_dict()

    def load_latency_state_dict(self, state):
        self._scheduler.load_state_dict(state)
