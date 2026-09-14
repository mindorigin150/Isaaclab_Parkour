"""RSL-RL vector adapter for Parkour latent command training."""

from __future__ import annotations

import torch

from training.common.action_latency import ActionLatencyBatch

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
        control_repeat: int = 5,
        gamma: float = 0.99,
        clip_actions: float | None = None,
    ):
        super().__init__(env, clip_actions=None)
        self.clip_actions = clip_actions
        self.decoder = decoder.eval().requires_grad_(False)
        self.control_repeat = control_repeat
        self.gamma = gamma
        self.command_dim = latency_config["command"]["dimension"]
        self.num_actions = self.command_dim
        self._scheduler = ActionLatencyBatch(
            latency_config,
            num_envs=self.num_envs,
            device=self.device,
            noop_command=torch.zeros(self.command_dim, device=self.device),
        )
        self._scheduler.reset()
        self._raw_obs, _ = self.get_observations()
        self._actor_obs = torch.empty_like(self._raw_obs)
        self._raw_step_budget = torch.full(
            (self.num_envs,), self.control_repeat, dtype=torch.long, device=self.device
        )
        self._gamma = torch.full((self.num_envs,), self.gamma, device=self.device)

    def step(self, commands: torch.Tensor, raw_step_budget: torch.Tensor | None = None):
        if self.clip_actions is not None:
            commands = torch.clamp(commands, -self.clip_actions, self.clip_actions)
        commands = commands.reshape(self.num_envs, self.command_dim)
        full_batch = raw_step_budget is None
        if full_batch:
            raw_step_budget = self._raw_step_budget
            budgets = [self.control_repeat] * self.num_envs
        else:
            raw_step_budget = raw_step_budget.to(device=self.device, dtype=torch.long)
            budgets = raw_step_budget.tolist()
        reward = torch.zeros(self.num_envs, device=self.device)
        raw_reward = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        truncation = torch.zeros_like(dones)
        executed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        terminal_obs = None
        extras = {}
        active = raw_step_budget > 0
        step_ids = None if full_batch else [env_id for env_id, budget in enumerate(budgets) if budget > 0]
        admission = self._scheduler.submit(commands, env_ids=step_ids)
        done_ids = []
        for raw_frame in range(self.control_repeat):
            step_mask = active & (executed_steps < raw_step_budget)
            if step_ids is not None and not step_ids:
                break
            applied = self._scheduler.actions(env_ids=step_ids)
            latent = applied[:, :32]
            yaw = applied[:, 32:] * GO2_PARKOUR_YAW_SCALE
            actor_obs = self._actor_obs
            actor_obs.copy_(self._raw_obs)
            actor_obs[:, 6:8] = yaw
            with torch.inference_mode():
                motor_action = self.decoder(
                    actor_obs,
                    hist_encoding=True,
                    scandots_latent=latent,
                )
            obs_dict, frame_reward, terminated, truncated, extras = self.unwrapped.step_no_reset(
                motor_action, active_mask=step_mask
            )
            self._raw_obs = obs_dict["policy"]
            frame_done = terminated | truncated
            frame_discount = self.gamma**raw_frame
            masked_reward = torch.where(step_mask, frame_reward, 0.0)
            reward += frame_discount * masked_reward
            raw_reward += masked_reward
            executed_steps += step_mask
            dones |= frame_done & step_mask
            truncation |= truncated & step_mask
            newly_done = step_mask & frame_done
            # The CPU scheduler needs done flags once; reuse them for active and reset IDs.
            done_flags = newly_done.tolist()
            current_ids = range(self.num_envs) if step_ids is None else step_ids
            new_done_ids = [env_id for env_id in current_ids if done_flags[env_id]]
            done_ids.extend(new_done_ids)
            if new_done_ids:
                if terminal_obs is None:
                    terminal_obs = {key: value.clone() for key, value in obs_dict.items()}
                else:
                    for key, value in obs_dict.items():
                        terminal = terminal_obs[key]
                        mask = newly_done.reshape((-1,) + (1,) * (value.ndim - 1))
                        torch.where(mask, value, terminal, out=terminal)
            active &= ~frame_done
            self._scheduler.advance(env_ids=step_ids)
            step_ids = [
                env_id for env_id in current_ids
                if not done_flags[env_id] and raw_frame + 1 < budgets[env_id]
            ]
        if terminal_obs is not None:
            extras["terminal_observations"] = terminal_obs
        extras["latency_executed_steps"] = executed_steps
        extras["latency_raw_reward"] = raw_reward
        extras["latency_active"] = raw_step_budget > 0
        extras["latency_discount"] = torch.pow(self._gamma, executed_steps)
        extras["latency_admission"] = admission
        extras["latency_sampled_ms"] = torch.tensor(
            [
                submission["latency_ms"] if submission is not None else float("nan")
                for submission in self._scheduler.last_submission
            ],
            dtype=torch.float32,
            device=self.device,
        )
        done_ids.sort()
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
