# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect and evaluate the fixed-controller Extreme Parkour VLA benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import gymnasium as gym

PARKOUR_TASK = "Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0"
PARKOUR_EVAL_MODES = {"teacher-eval", "oracle-eval", "latency-eval"}
PARKOUR_ACTOR_OBSERVATION_DIM = 753
REPO_ROOT = Path(__file__).resolve().parents[4]
PARKOUR_REPO_ROOT = Path(__file__).resolve().parents[2]
PARKOUR_TASKS_ROOT = PARKOUR_REPO_ROOT / "parkour_tasks"
for import_root in map(
    str,
    (REPO_ROOT, PARKOUR_REPO_ROOT, PARKOUR_TASKS_ROOT),
):
    if import_root in sys.path:
        sys.path.remove(import_root)
sys.path[:0] = [str(PARKOUR_TASKS_ROOT), str(PARKOUR_REPO_ROOT), str(REPO_ROOT)]

from latency_bench.core.latency_distribution import derive_seed
from latency_bench.core.config import load_config
from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.data.parkour_dagger import PARKOUR_ACTION_HORIZON
from latency_bench.data.dagger import DaggerStep, collect_dagger
from latency_bench.data.episode_io import encode_video
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.executors.env_step_backend import EnvStepResponse, RemoteEnvSlotHandle

if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    import cli_args  # isort: skip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "teacher-eval",
            "oracle-eval",
            "collect",
            "dagger-collect",
            "latency-eval",
        ),
    )
    parser.add_argument("--task", default=PARKOUR_TASK)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--disable_fabric", action="store_true")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true")
    parser.add_argument(
        "--command-checkpoint",
        type=Path,
        help="Native latency-command actor used by collection; --checkpoint remains the decoder.",
    )
    parser.add_argument("--latency-config", type=Path)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--train_episodes", type=int, default=250)
    parser.add_argument("--val_episodes", type=int, default=25)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--policy_config", type=Path)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--inference_device", action="append")
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--dagger_round", type=int, default=0)
    parser.add_argument("--dagger_row_budget", type=int, default=64_000)
    parser.add_argument("--dagger_shard_rows", type=int, default=1_000)
    parser.add_argument("--action_horizon", type=int, default=40)
    parser.add_argument("--startup-ready-file", type=Path)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.inference_device is None:
        args_cli.inference_device = ["cuda:1"]
    if args_cli.num_envs is None:
        args_cli.num_envs = (
            192
            if args_cli.mode == "dagger-collect"
            else 50
        )
    if args_cli.mode == "latency-eval":
        args_cli.headless = True
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import omni.usd
    import torch
    import usdrt

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import isaaclab_tasks  # noqa: F401
    from parkour_isaaclab.actor import (
        GO2_PARKOUR_MTS_THRESHOLD_RAD,
        GO2_PARKOUR_YAW_SCALE,
        apply_parkour_mts,
    )
    from parkour_tasks.extreme_parkour_task.config.go2.parkour_vla_cfg import (
        PARKOUR_VLA_LATENT_DIM,
        PARKOUR_VLA_ACTION_DIM,
        PARKOUR_VLA_CONTROL_REPEAT,
        PARKOUR_VLA_PROMPT,
        PARKOUR_VLA_PROPRIO_DIM,
        PARKOUR_VLA_YAW_DIM,
    )
    from scripts.rsl_rl.modules.on_policy_runner_with_extractor import (
        OnPolicyRunnerWithExtractor,
    )
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper


def _load_environment_and_teacher():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if args_cli.mode in PARKOUR_EVAL_MODES:
        env_cfg.episode_length_s = (
            args_cli.max_episode_steps * env_cfg.sim.dt * env_cfg.decimation
        )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            str(log_root.resolve()), agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.mode in PARKOUR_EVAL_MODES:
        env.unwrapped.scene.terrain.cfg.terrain_generator.curriculum = False
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    decoder_runner = OnPolicyRunnerWithExtractor(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    decoder_runner.load(resume_path, load_optimizer=False)
    decoder_policy = decoder_runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.command_checkpoint is None:
        return env, decoder_runner.alg.policy.actor, decoder_policy, Path(resume_path)

    command_env = ParkourRslRlVecEnvWrapper(
        env, clip_actions=agent_cfg.clip_actions
    )
    command_env.num_actions = PARKOUR_ACTION_HORIZON * PARKOUR_VLA_ACTION_DIM
    command_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    command_cfg.policy.actor.class_name = "CommandActor"
    command_cfg.policy.actor.action_horizon = PARKOUR_ACTION_HORIZON
    command_cfg.policy.actor.command_dim = PARKOUR_VLA_ACTION_DIM
    command_runner = OnPolicyRunnerWithExtractor(
        command_env, command_cfg.to_dict(), log_dir=None, device=command_cfg.device
    )
    command_runner.load(args_cli.command_checkpoint, load_optimizer=False)
    return (
        env,
        command_runner.get_inference_policy(device=env.unwrapped.device),
        decoder_policy,
        Path(resume_path),
    )


def _parkour_actor_action(env, actor, observation, action_value):
    vla_action = torch.as_tensor(
        action_value, device=env.device, dtype=observation.dtype
    ).reshape(-1, PARKOUR_VLA_LATENT_DIM + PARKOUR_VLA_YAW_DIM)
    latent = vla_action[:, :PARKOUR_VLA_LATENT_DIM]
    predicted_yaw = vla_action[:, PARKOUR_VLA_LATENT_DIM:]
    actor_observation = _actor_observation_with_yaw(observation, predicted_yaw)
    with torch.inference_mode():
        return actor(
            actor_observation,
            hist_encoding=True,
            scandots_latent=latent,
        )


class ParkourEnvStepBackend:
    """Expose the Isaac Lab vector scene through the common eval contract."""

    backend_name = "isaaclab_parkour"

    def __init__(self, env, actor, *, noop_action: Action):
        self._env = env
        self._raw_env = env.unwrapped
        self._actor = actor
        self._num_slots = int(env.num_envs)
        self._env_fps = 1.0 / float(self._raw_env.step_dt)
        self._frame_ms = 1000.0 / self._env_fps
        self.noop_action = noop_action
        self.closed = False
        self.last_step_metadata_by_slot: dict[int, dict[str, object]] = {}
        self._obs, _ = env.get_observations()
        self._edge_term = self._raw_env.reward_manager.get_term_cfg(
            "reward_feet_edge"
        ).func
        self._parkour = self._raw_env.parkour_manager.get_term("base_parkour")
        self._edge_sum = torch.zeros(self._num_slots, dtype=torch.float64)
        self._edge_sq_sum = torch.zeros(self._num_slots, dtype=torch.float64)
        self._edge_steps = torch.zeros(self._num_slots, dtype=torch.int64)
        self._last_motor_action: torch.Tensor | None = None
        self.slot_handles = [
            RemoteEnvSlotHandle(
                slot_id=slot_id,
                env_fps=self._env_fps,
                noop_action=noop_action,
                action_space=gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(PARKOUR_VLA_LATENT_DIM + PARKOUR_VLA_YAW_DIM,),
                    dtype=np.float32,
                ),
            )
            for slot_id in range(self._num_slots)
        ]

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def env_fps(self) -> float:
        return self._env_fps

    def worker_pids(self) -> list[int]:
        return []

    def reset_slot(
        self, slot_id: int, *, episode_id: int, seed: int | None
    ) -> Observation:
        self._raise_if_closed()
        slot_id = int(slot_id)
        env_ids = torch.tensor(
            [slot_id], dtype=torch.int64, device=self._raw_env.device
        )
        obs_dict, _ = self._raw_env.reset(seed=seed, env_ids=env_ids)
        self._obs = obs_dict["policy"]
        if self._last_motor_action is not None:
            self._last_motor_action[slot_id] = 0.0
        self._edge_sum[slot_id] = 0.0
        self._edge_sq_sum[slot_id] = 0.0
        self._edge_steps[slot_id] = 0
        observation = self._make_observation(slot_id, episode_id=episode_id)
        self.slot_handles[slot_id].update_observation(observation)
        return observation

    def observe_slots(self, slot_ids: Sequence[int]) -> dict[int, Observation]:
        self._raise_if_closed()
        rgb = _rgb_frames(self._env)
        observations = {
            int(slot_id): self._make_observation(int(slot_id), rgb=rgb)
            for slot_id in slot_ids
        }
        for slot_id, observation in observations.items():
            self.slot_handles[slot_id].update_observation(observation)
        return observations

    def step_slots(
        self, actions_by_slot: Mapping[int, Action]
    ) -> dict[int, EnvStepResponse]:
        self._raise_if_closed()
        start = time.perf_counter()
        slot_ids = [int(slot_id) for slot_id in actions_by_slot]
        goal_index = self._parkour.cur_goal_idx.clone()
        active_mask = torch.zeros(self._num_slots, dtype=torch.bool, device=self._obs.device)
        active_mask[slot_ids] = True
        vla_action = torch.zeros(
            (self._num_slots, PARKOUR_VLA_LATENT_DIM + PARKOUR_VLA_YAW_DIM),
            dtype=self._obs.dtype,
            device=self._obs.device,
        )
        for slot_id, action in actions_by_slot.items():
            vla_action[int(slot_id)] = torch.as_tensor(
                np.asarray(action.value, dtype=np.float32),
                dtype=self._obs.dtype,
                device=self._obs.device,
            )
        motor_action = _parkour_actor_action(
            self._env, self._actor, self._obs, vla_action
        )
        if self._last_motor_action is not None:
            motor_action = motor_action.clone()
            motor_action[~active_mask] = self._last_motor_action[~active_mask]
        self._last_motor_action = motor_action.detach().clone()
        obs_dict, rewards, terminated, truncated, _extras = self._raw_env.step_no_reset(
            motor_action, active_mask=active_mask
        )
        self._obs = obs_dict["policy"]
        successes = self._raw_env.termination_manager.get_term("parkour_success")
        edge = self._edge_term.feet_at_edge.sum(dim=1).to(dtype=torch.float64)
        for slot_id in slot_ids:
            value = float(edge[slot_id])
            self._edge_sum[slot_id] += value
            self._edge_sq_sum[slot_id] += value * value
            self._edge_steps[slot_id] += 1

        elapsed = time.perf_counter() - start
        responses: dict[int, EnvStepResponse] = {}
        for slot_id in slot_ids:
            slot_id = int(slot_id)
            count = int(self._edge_steps[slot_id])
            progress = float(
                (goal_index[slot_id] + successes[slot_id].long()).float()
                / self._parkour.num_goals
            )
            info = {
                "task_metrics": {
                    "normalized_waypoint_progress": progress,
                    "edge_violation": float(self._edge_sum[slot_id]) / count,
                },
                "task_metric_moments": {
                    "edge_violation": {
                        "sum": float(self._edge_sum[slot_id]),
                        "sum_sq": float(self._edge_sq_sum[slot_id]),
                        "count": count,
                    }
                },
                "env_step": int(self._raw_env.episode_length_buf[slot_id]),
                "sim_time_ms": float(self._raw_env.episode_length_buf[slot_id])
                * self._frame_ms,
                "applied_action": np.asarray(actions_by_slot[slot_id].value).tolist(),
                "applied_action_name": actions_by_slot[slot_id].name,
            }
            result = StepResult(
                observation=None,
                reward=float(rewards[slot_id]),
                done=bool(terminated[slot_id]),
                truncated=bool(truncated[slot_id]),
                info=info,
            )
            metadata = {
                "backend": self.backend_name,
                "slot_id": slot_id,
                "worker_pid": None,
                "worker_step_wall_sec": elapsed,
            }
            self.last_step_metadata_by_slot[slot_id] = metadata
            responses[slot_id] = EnvStepResponse(
                result=result,
                worker_pid=None,
                metadata=metadata,
            )
        return responses

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._env.close()

    def _make_observation(
        self,
        slot_id: int,
        *,
        episode_id: int | None = None,
        rgb: np.ndarray | None = None,
    ) -> Observation:
        if rgb is None:
            rgb = _rgb_frames(self._env)
        state = _masked_vla_state(
            self._obs[:, :PARKOUR_VLA_PROPRIO_DIM].detach().cpu().numpy()
        )
        metadata = {
            ENV_RAW_RGB_FRAME_STACK_INFO_KEY: rgb[slot_id][None],
            "parkour_proprio": state[slot_id],
            "slot_id": slot_id,
        }
        if episode_id is not None:
            metadata["episode_id"] = int(episode_id)
        env_step = int(self._raw_env.episode_length_buf[slot_id])
        return Observation(
            data=None,
            env_step=env_step,
            sim_time_ms=env_step * self._frame_ms,
            metadata=metadata,
        )

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("env step backend is closed")


def build_latency_eval_backend(config: dict, env, actor) -> ParkourEnvStepBackend:
    noop_action = Action(
        value=np.asarray(config["env"]["noop_action"], dtype=np.float32),
        name="noop",
        is_noop=True,
    )
    return ParkourEnvStepBackend(env, actor, noop_action=noop_action)


def _run_latency_eval(eval_config: dict, env, actor) -> dict:
    from latency_bench.eval.driver import run_from_config

    backend = build_latency_eval_backend(eval_config, env, actor)
    run_from_config(
        eval_config,
        env_backend=backend,
        inference_devices=eval_config["executor"]["inference_devices"],
    )
    return {"output_dir": eval_config["logging"]["output_dir"]}


def _rgb_frames(env) -> np.ndarray:
    camera = env.unwrapped.scene["vla_camera"]
    visible = camera.data.output["rgb"][..., :3].clone()
    instance_ids = camera.data.output["instance_segmentation_fast"][..., 0].clone()
    id_to_labels = camera.data.info["instance_segmentation_fast"]["idToLabels"].copy()

    stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    # Parkour's only Fabric mesh prototypes are the 17 shared Go2 visuals.
    visibility_attributes = [
        stage.GetPrimAtPath(path).CreateAttribute(
            "_worldVisibility", usdrt.Sdf.ValueTypeNames.Bool, False
        )
        for path in stage.GetPrimsWithTypeName("Mesh")
        if str(path).startswith("/__Prototype_")
    ]
    for attribute in visibility_attributes:
        attribute.Set(False)
    try:
        env.unwrapped.sim.render()
        camera._is_outdated[:] = True
        camera.update(0.0, force_recompute=True)
        background = camera.data.output["rgb"][..., :3].clone()
    finally:
        for attribute in visibility_attributes:
            attribute.Set(True)

    own_robot_pixels = torch.zeros_like(instance_ids, dtype=torch.bool)
    for instance_id, robot_path in id_to_labels.items():
        if robot_path.startswith("/World/envs/"):
            env_id = int(robot_path.split("/")[3].removeprefix("env_"))
            own_robot_pixels[env_id] |= instance_ids[env_id] == instance_id
    background[own_robot_pixels] = visible[own_robot_pixels]
    return background.detach().cpu().numpy()


def _policy_observations(
    rgb: np.ndarray,
    state: np.ndarray,
    slots: list[int],
    action_noise_seeds: Sequence[int],
    step: int,
):
    from latency_bench.core.types import Observation
    from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY

    return [
        Observation(
            # Official GR00T consumes the frame stack from metadata. Keeping
            # the legacy data carrier would pickle every RGB frame twice.
            data=None,
            env_step=step,
            sim_time_ms=step * 20.0,
            metadata={
                ENV_RAW_RGB_FRAME_STACK_INFO_KEY: rgb[slot][None],
                "parkour_proprio": state[slot],
                "slot_id": slot,
                "action_noise_seed": action_noise_seeds[slot],
            },
        )
        for slot in slots
    ]


def _masked_vla_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).copy()
    state[:, 6:8] = 0.0
    return state


def _new_policy_pool():
    from latency_bench.executors.realtime.pool import ProcessInferencePool

    config = load_config(args_cli.policy_config)
    config["executor"]["inference_batch_size"] = args_cli.inference_batch_size
    return ProcessInferencePool(config=config, inference_devices=args_cli.inference_device)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def _actor_observation_with_yaw(obs, normalized_yaw):
    actor_obs = obs.clone()
    actor_obs[:, 6:8] = normalized_yaw * GO2_PARKOUR_YAW_SCALE
    return actor_obs


def _evaluate(env, actor, teacher_policy, *, use_oracle: bool) -> dict:
    """Evaluate the native teacher or privileged oracle controller."""
    obs, _ = env.get_observations()
    num_envs = env.num_envs
    latent = torch.zeros(
        (num_envs, PARKOUR_VLA_LATENT_DIM), dtype=obs.dtype, device=env.device
    )
    returns = torch.zeros(num_envs, dtype=torch.float, device=env.device)
    lengths = torch.zeros(num_envs, dtype=torch.float, device=env.device)
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    progress: list[float] = []
    edge_violations: list[float] = []
    episode_edges: list[list[float]] = [[] for _ in range(num_envs)]
    edge_term = env.unwrapped.reward_manager.get_term_cfg("reward_feet_edge").func
    parkour = env.unwrapped.parkour_manager.get_term("base_parkour")

    while len(episode_returns) < args_cli.eval_episodes:
        with torch.inference_mode():
            if use_oracle:
                latent = actor.infer_scandots_latent(obs)
                actions = teacher_policy(
                    _actor_observation_with_yaw(
                        obs, torch.zeros(num_envs, 2, device=obs.device)
                    ),
                    hist_encoding=True,
                    scandots_latent=latent,
                )
            else:
                actions = teacher_policy(obs, hist_encoding=True)

        goal_index = parkour.cur_goal_idx.clone()
        obs, rewards, dones, _ = env.step(actions)
        successes = env.unwrapped.termination_manager.get_term("parkour_success")
        edge = edge_term.feet_at_edge.sum(dim=1).float()
        for slot, value in enumerate(edge.cpu().numpy().tolist()):
            episode_edges[slot].append(value)
        returns += rewards
        lengths += 1

        done_ids = dones.bool().nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            remaining = args_cli.eval_episodes - len(episode_returns)
            accepted_ids = done_ids[:remaining]
            episode_returns.extend(returns[accepted_ids].cpu().numpy().tolist())
            episode_lengths.extend(lengths[accepted_ids].cpu().numpy().tolist())
            progress.extend(
                (
                    (goal_index[accepted_ids] + successes[accepted_ids].long()).float()
                    / parkour.num_goals
                )
                .cpu()
                .numpy()
                .tolist()
            )
            for slot in accepted_ids.cpu().tolist():
                edge_violations.extend(episode_edges[slot])
            returns[done_ids] = 0
            lengths[done_ids] = 0
            for slot in done_ids.cpu().tolist():
                episode_edges[slot].clear()

    return {
        "episodes": len(episode_returns),
        "max_episode_steps": args_cli.max_episode_steps,
        "reward": _summary(episode_returns),
        "episode_length": _summary(episode_lengths),
        "normalized_waypoint_progress": _summary(progress),
        "edge_violation": _summary(edge_violations),
    }


def _collect(env, actor, teacher_policy, checkpoint: Path) -> dict:
    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    episode_target = args_cli.train_episodes + args_cli.val_episodes
    split_order = ["train"] * args_cli.train_episodes + ["val"] * args_cli.val_episodes
    random.Random(args_cli.split_seed).shuffle(split_order)

    obs, _ = env.get_observations()
    num_envs = env.num_envs
    phase = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    latent = torch.zeros(
        (num_envs, PARKOUR_VLA_LATENT_DIM), dtype=obs.dtype, device=env.device
    )
    commands = torch.zeros(
        (num_envs, PARKOUR_VLA_ACTION_DIM), dtype=obs.dtype, device=env.device
    )
    block_terminated = [False] * num_envs
    block_success = [False] * num_envs
    rows_by_slot: list[list[dict]] = [[] for _ in range(num_envs)]
    accepted = 0
    control_step = 0

    while accepted < episode_target and simulation_app.is_running():
        due_ids = (phase == 0).nonzero(as_tuple=False).flatten()
        obs_cpu = obs.detach().cpu().numpy()
        due_slots = due_ids.cpu().tolist()
        if due_slots:
            rgb = _rgb_frames(env)
            with torch.inference_mode():
                latent[due_ids] = actor.infer_scandots_latent(obs[due_ids])
                commands[due_ids] = torch.cat(
                    (
                        latent[due_ids],
                        obs[due_ids, 6:8] / GO2_PARKOUR_YAW_SCALE,
                    ),
                    dim=1,
                )
            due_actions = commands[due_ids].detach().cpu().numpy()
            due_states = _masked_vla_state(
                obs_cpu[due_slots, :PARKOUR_VLA_PROPRIO_DIM]
            )
            for slot, state, action in zip(due_slots, due_states, due_actions):
                block_terminated[slot] = False
                block_success[slot] = False
                rows_by_slot[slot].append(
                    {
                        "rgb": rgb[slot],
                        "observation.state": state,
                        "action": action,
                        "actor_observation": [],
                        "termination": [],
                        "raw_reward": 0.0,
                    }
                )

        for slot in range(num_envs):
            rows_by_slot[slot][-1]["actor_observation"].append(obs_cpu[slot].copy())
        with torch.inference_mode():
            actor_observation = _actor_observation_with_yaw(
                obs, commands[:, PARKOUR_VLA_LATENT_DIM:]
            )
            actions = teacher_policy(
                actor_observation,
                hist_encoding=True,
                scandots_latent=commands[:, :PARKOUR_VLA_LATENT_DIM],
            )
        obs, rewards, dones, _ = env.step(actions)
        successes = env.unwrapped.termination_manager.get_term("parkour_success")
        rewards_cpu = rewards.detach().cpu().numpy()
        dones_cpu = dones.detach().cpu().numpy().astype(bool)
        successes_cpu = successes.detach().cpu().numpy().astype(bool)
        for slot, row in enumerate(rows_by_slot):
            row[-1]["raw_reward"] += float(rewards_cpu[slot])
            row[-1]["termination"].append(bool(dones_cpu[slot]))
        phase = (phase + 1) % PARKOUR_VLA_CONTROL_REPEAT
        control_step += 1

        for slot in np.flatnonzero(dones_cpu).tolist():
            block_terminated[slot] = True
            block_success[slot] |= bool(successes_cpu[slot])

        for slot in (phase == 0).nonzero(as_tuple=False).flatten().cpu().tolist():
            if not block_terminated[slot]:
                continue
            if (block_success[slot] or args_cli.keep_failed) and accepted < episode_target:
                split = split_order[accepted]
                _write_dagger_shard(
                    output_dir,
                    accepted,
                    rows_by_slot[slot],
                    split=split,
                )
                accepted += 1
                print(f"accepted {accepted}/{episode_target}: split={split} slot={slot}")
            rows_by_slot[slot] = []
            block_terminated[slot] = False
            block_success[slot] = False

    metadata = {
        "schema_version": 4,
        "env_name": "extreme_parkour_go2",
        "integration_name": "isaaclab_parkour",
        "action_layout": "go2_parkour_terrain_latent_yaw_v1",
        "action_labels": [
            *[
                f"terrain_latent_{index}" for index in range(PARKOUR_VLA_LATENT_DIM)
            ],
            "yaw_current",
            "yaw_next",
        ],
        "state_labels": [
            f"parkour_proprio_{index}" for index in range(PARKOUR_VLA_PROPRIO_DIM)
        ],
        "reward_field": "raw_reward",
        "rows_unit": "decision_step",
        "env_fps": 50,
        "obs_fps": 10,
        "obs_stride_raw_frames": 5,
        "base_prompt": PARKOUR_VLA_PROMPT,
        "teacher_checkpoint": str(checkpoint),
        "train_episodes": args_cli.train_episodes,
        "val_episodes": args_cli.val_episodes,
        "split_seed": args_cli.split_seed,
        "control_steps": control_step,
        "control_repeat": PARKOUR_VLA_CONTROL_REPEAT,
        "actor_observation_shape": [
            PARKOUR_VLA_CONTROL_REPEAT,
            PARKOUR_ACTOR_OBSERVATION_DIM,
        ],
        "shard_format": "npz+mp4",
        "shard_root": "rollout_shards",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _collect_latency(env, command_policy, decoder_policy, checkpoint: Path) -> dict:
    # Isaac's app must be initialized before importing the tensor training adapter.
    from training.common.action_latency import ActionLatencyBatch

    config = load_config(args_cli.latency_config)
    output = args_cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    splits = ["train"] * args_cli.train_episodes + ["val"] * args_cli.val_episodes
    random.Random(args_cli.split_seed).shuffle(splits)
    transport = ActionLatencyBatch(
        config, num_envs=env.num_envs, device=env.device,
        noop_command=torch.zeros(PARKOUR_VLA_ACTION_DIM, device=env.device),
    )
    transport.reset()
    obs, _ = env.get_observations()
    commands = torch.zeros(
        env.num_envs, PARKOUR_ACTION_HORIZON, PARKOUR_VLA_ACTION_DIM,
        device=env.device, dtype=obs.dtype,
    )
    rows = [[] for _ in range(env.num_envs)]
    actor_inputs = [[] for _ in range(env.num_envs)]
    terminations = [[] for _ in range(env.num_envs)]
    control_traces = [[] for _ in range(env.num_envs)]
    accepted = 0
    completed = 0
    raw_steps = 0
    while accepted < len(splits):
        due = [slot for slot, frame in enumerate(transport.frames)
               if frame % transport.clock.obs_stride_raw_frames == 0]
        if due:
            with torch.inference_mode():
                commands[due] = command_policy(obs[due]).reshape(
                    len(due), PARKOUR_ACTION_HORIZON, PARKOUR_VLA_ACTION_DIM
                )
                if env.clip_actions is not None:
                    commands[due] = commands[due].clamp(-env.clip_actions, env.clip_actions)
            admitted = transport.submit(commands, env_ids=due).nonzero().flatten().tolist()
            if admitted:
                rgb = _rgb_frames(env)
                states = _masked_vla_state(
                    obs[admitted, :PARKOUR_VLA_PROPRIO_DIM].cpu().numpy()
                )
                for slot, state in zip(admitted, states):
                    rows[slot].append({
                        "rgb": rgb[slot], "observation.state": state,
                        "action": commands[slot].cpu().numpy().copy(),
                        "raw_reward": 0.0, **transport.last_submission[slot],
                    })
        before_step = obs.cpu().numpy()
        for slot in range(env.num_envs):
            actor_inputs[slot].append(before_step[slot].copy())
        applied = transport.actions()
        applied_cpu = applied.cpu().numpy()
        motor_actions = _parkour_actor_action(env, decoder_policy, obs, applied)
        obs_dict, rewards, terminated, truncated, _ = env.unwrapped.step_no_reset(motor_actions)
        obs = obs_dict["policy"]
        done = terminated | truncated
        success = env.unwrapped.termination_manager.get_term("parkour_success")
        for slot in range(env.num_envs):
            terminations[slot].append(done[slot].item())
            rows[slot][-1]["raw_reward"] += rewards[slot].item()
            control_traces[slot].append({
                "applied_command": applied_cpu[slot].copy(),
                **transport.last_application[slot],
                "reward": rewards[slot].item(), "done": done[slot].item(),
            })
        transport.advance()
        raw_steps += env.num_envs
        done_ids = done.nonzero().flatten().tolist()
        for slot in done_ids:
            completed += 1
            if accepted < len(splits) and (args_cli.keep_failed or success[slot].item()):
                observations = np.stack(actor_inputs[slot])
                terminal = np.asarray(terminations[slot], dtype=bool)
                for row in rows[slot]:
                    start = row["issued_raw_frame"]
                    aligned = observations[start:start + PARKOUR_ACTION_HORIZON]
                    mask = terminal[start:start + PARKOUR_ACTION_HORIZON]
                    padding = PARKOUR_ACTION_HORIZON - len(aligned)
                    row["actor_observation"] = np.pad(aligned, ((0, padding), (0, 0)), mode="edge")
                    row["termination"] = np.pad(mask, (0, padding), constant_values=True)
                control_trace = {
                    f"control_{name}": np.asarray([frame[name] for frame in control_traces[slot]])
                    for name in control_traces[slot][0]
                }
                _write_dagger_shard(
                    output, accepted, rows[slot], split=splits[accepted],
                    control_trace=control_trace,
                )
                accepted += 1
                print(f"accepted {accepted}/{len(splits)}: slot={slot} raw_frames={len(terminal)}", flush=True)
            rows[slot] = []
            actor_inputs[slot] = []
            terminations[slot] = []
            control_traces[slot] = []
        if done_ids:
            reset_obs, _ = env.unwrapped.reset(env_ids=torch.tensor(done_ids, device=env.device))
            obs = reset_obs["policy"]
            transport.reset(done_ids)
    metadata = {
        "schema_version": 7, "env_name": "extreme_parkour_go2",
        "rows_unit": "admitted_observation", "env_fps": config["env"]["env_fps"],
        "obs_fps": config["env"]["obs_fps"],
        "issued_command_shape": [PARKOUR_ACTION_HORIZON, PARKOUR_VLA_ACTION_DIM],
        "actor_observation_shape": [PARKOUR_ACTION_HORIZON, PARKOUR_ACTOR_OBSERVATION_DIM],
        "termination_shape": [PARKOUR_ACTION_HORIZON], "train_episodes": args_cli.train_episodes,
        "val_episodes": args_cli.val_episodes, "completed_episodes": completed,
        "raw_steps": raw_steps, "keep_failed": args_cli.keep_failed,
        "command_checkpoint": str(args_cli.command_checkpoint.resolve()),
        "decoder_checkpoint": str(checkpoint.resolve()), "latency_config": config,
        "shard_format": "npz+mp4", "shard_root": "rollout_shards",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _write_dagger_shard(
    output_dir: Path, shard_index: int, rows: list[dict], *, split: str = "train",
    control_trace: dict | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix=".dagger_", dir=output_dir) as directory:
        video_tmp = Path(directory) / "episode.mp4"

        encode_video([row["rgb"] for row in rows], video_tmp, fps=10, executable="ffmpeg")

        payload = {
            "state": np.stack(
                [row["observation.state"] for row in rows]
            ).astype(np.float32),
            "action": np.stack([row["action"] for row in rows]).astype(np.float32),
            "termination": np.stack([row["termination"] for row in rows]).astype(bool),
            "actor_observation": np.stack(
                [row["actor_observation"] for row in rows]
            ).astype(np.float32),
            "image_shape": np.asarray(rows[0]["rgb"].shape, dtype=np.int64),
        }
        if "raw_reward" in rows[0]:
            payload["raw_reward"] = np.asarray(
                [row["raw_reward"] for row in rows], dtype=np.float32
            )
        if payload["action"].ndim == 3:
            for field in ("episode_id", "obs_id", "issued_raw_frame", "ready_raw_frame", "latency_ms", "worker_id"):
                payload[field] = np.asarray([row[field] for row in rows])
        if control_trace is not None:
            payload.update(control_trace)
        # Keep the video-backed writer at the collection boundary so Isaac Lab can
        # run without importing conversion-only dependencies.
        from latency_bench.data.parkour_dagger import write_parkour_dagger_shard

        write_parkour_dagger_shard(
            output_dir,
            split=split,
            episode_idx=shard_index,
            arrays=payload,
            video_path=video_tmp,
        )


def _existing_dagger_rows(output_dir: Path) -> int:
    shard_dir = output_dir / "rollout_shards" / "train"
    row_count = 0
    for path in sorted(shard_dir.glob("episode_*")):
        if not path.is_dir():
            continue
        shard_path = path / "episode.npz"
        with np.load(shard_path) as shard:
            row_count += int(shard["state"].shape[0])
    return row_count


class ParkourDaggerAdapter:
    """Retain native MTS, five-step teacher traces, and early-done padding."""

    budget_mode = "anchor"
    seed_mode = "slot"
    reset = None

    def __init__(self, env, actor, teacher_policy, output_dir):
        self.inference_period = self.record_period = PARKOUR_VLA_CONTROL_REPEAT
        self.env, self.actor, self.teacher_policy = env, actor, teacher_policy
        self.output_dir = output_dir
        self.num_envs = env.num_envs
        self.shard_rows = args_cli.dagger_shard_rows
        self.obs, _ = env.get_observations()
        self.mts_prediction_count = self.mts_total_count = 0

    def snapshot(self, due, record_slots):
        actor_obs = self.obs.detach().cpu().numpy().copy()
        return {
            "state_cpu": actor_obs[:, :PARKOUR_VLA_PROPRIO_DIM],
            "actor_obs": actor_obs,
            "rgb": _rgb_frames(self.env) if due else None,
        }

    def observations(self, snapshot, slots, state):
        return _policy_observations(
            snapshot["rgb"], _masked_vla_state(snapshot["state_cpu"]), slots,
            state.seeds, state.control_step,
        )

    def start_records(self, snapshot, slots, state):
        with torch.inference_mode():
            latent = self.actor.infer_scandots_latent(self.obs[slots]).cpu().numpy()
        actions = np.concatenate(
            (latent, snapshot["state_cpu"][slots, 6:8] / GO2_PARKOUR_YAW_SCALE), axis=1,
        )
        states = _masked_vla_state(snapshot["state_cpu"][slots])
        return [
            {"rgb": snapshot["rgb"][slot].copy(), "observation.state": value.copy(),
             "action": action.copy(), "actor_observation": [], "termination": []}
            for slot, value, action in zip(slots, states, actions, strict=True)
        ]

    def step(self, snapshot, active, state):
        vla_action = torch.from_numpy(np.stack([
            state.outputs[slot].action_chunk[
                min(state.episode_steps[slot] % self.inference_period, args_cli.action_horizon - 1)
            ]
            for slot in active
        ])).to(device=self.env.device, dtype=self.obs.dtype)
        with torch.inference_mode():
            actor_observation, use_prediction = apply_parkour_mts(
                self.obs, vla_action[:, PARKOUR_VLA_LATENT_DIM:],
            )
            self.mts_prediction_count += use_prediction.sum().item()
            self.mts_total_count += use_prediction.numel()
            actions = self.teacher_policy(
                actor_observation, hist_encoding=True,
                scandots_latent=vla_action[:, :PARKOUR_VLA_LATENT_DIM],
            )
        self.obs, _, dones, _ = self.env.step(actions)
        return DaggerStep(dones.bool().nonzero(as_tuple=False).flatten().cpu().tolist(), None)

    def finish_record(self, row, snapshot, result, slot, finished):
        row["actor_observation"].append(snapshot["actor_obs"][slot].copy())
        terminated = slot in result.done
        row["termination"].append(terminated)
        if terminated:
            while len(row["termination"]) < self.record_period:
                row["actor_observation"].append(row["actor_observation"][-1].copy())
                row["termination"].append(True)
        if finished:
            row["actor_observation"] = np.asarray(row["actor_observation"], dtype=np.float32)
            row["termination"] = np.asarray(row["termination"], dtype=bool)

    def write_rows(self, shard_index, slot, rows, writer):
        rows[-1]["termination"][-1] = True
        writer.submit(_write_dagger_shard, self.output_dir, shard_index, rows)


def _collect_dagger(env, actor, teacher_policy, checkpoint: Path) -> dict:
    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = _existing_dagger_rows(output_dir)
    target_rows = args_cli.dagger_row_budget
    if row_count >= target_rows:
        return {
            "schema_version": 6,
            "round": args_cli.dagger_round,
            "rows": row_count,
            "control_steps": row_count * PARKOUR_VLA_CONTROL_REPEAT,
            "already_complete": True,
        }
    if row_count:
        raise RuntimeError(
            "A partial DAgger round cannot be resumed exactly because Isaac Lab "
            "simulator and RNG state are not checkpointed; recollect this round "
            "from an empty output directory."
        )

    adapter = ParkourDaggerAdapter(env, actor, teacher_policy, output_dir)
    row_count = collect_dagger(
        adapter, _new_policy_pool(), row_budget=target_rows, seed=args_cli.seed,
        is_running=simulation_app.is_running,
    )

    metadata = {
        "schema_version": 6,
        "env_name": "extreme_parkour_go2",
        "integration_name": "isaaclab_parkour",
        "round": args_cli.dagger_round,
        "action_layout": "go2_parkour_terrain_latent_yaw_v1",
        "action_labels": [
            *[
                f"terrain_latent_{index}" for index in range(PARKOUR_VLA_LATENT_DIM)
            ],
            "yaw_current",
            "yaw_next",
        ],
        "row_budget": target_rows,
        "rows": row_count,
        "control_budget": target_rows * PARKOUR_VLA_CONTROL_REPEAT,
        "control_steps": row_count * PARKOUR_VLA_CONTROL_REPEAT,
        "control_repeat": PARKOUR_VLA_CONTROL_REPEAT,
        "shard_format": "npz+mp4",
        "shard_root": "rollout_shards/train",
        "npz_state_key": "state",
        "env_fps": 50,
        "obs_fps": 10,
        "obs_stride_raw_frames": PARKOUR_VLA_CONTROL_REPEAT,
        "vla_action_dim": PARKOUR_VLA_ACTION_DIM,
        "vla_latent_dim": PARKOUR_VLA_LATENT_DIM,
        "vla_yaw_dim": PARKOUR_VLA_YAW_DIM,
        "vla_yaw_scale": GO2_PARKOUR_YAW_SCALE,
        "vla_proprio_dim": PARKOUR_VLA_PROPRIO_DIM,
        "actor_observation_shape": [
            PARKOUR_VLA_CONTROL_REPEAT,
            PARKOUR_ACTOR_OBSERVATION_DIM,
        ],
        "base_prompt": PARKOUR_VLA_PROMPT,
        "teacher_checkpoint": str(checkpoint),
        "failure_and_timeout_rows_retained": True,
        "state_yaw_indices_masked": [6, 7],
        "dagger_mts_threshold_rad": GO2_PARKOUR_MTS_THRESHOLD_RAD,
        "dagger_mts_prediction_fraction": adapter.mts_prediction_count / adapter.mts_total_count,
    }
    temporary = output_dir / "metadata.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_dir / "metadata.json")
    return metadata


def main() -> None:
    if args_cli.mode == "latency-eval":
        args_cli.eval_config = load_config(args_cli.eval_config)
        args_cli.checkpoint = args_cli.eval_config["env"]["runtime_checkpoint_path"]
        args_cli.num_envs = args_cli.eval_config["evaluation"]["eval_parallel_envs"]
        args_cli.eval_episodes = args_cli.eval_config["evaluation"]["eval_episodes"]
        args_cli.max_episode_steps = args_cli.eval_config["evaluation"]["eval_max_steps"]
        args_cli.seed = args_cli.eval_config["experiment"]["seed"]
        args_cli.device = args_cli.eval_config["env"]["simulator_device"]
    exit_code = 0
    env = None
    try:
        env, actor, teacher_policy, checkpoint = _load_environment_and_teacher()
        if args_cli.startup_ready_file is not None:
            args_cli.startup_ready_file.touch()
        if args_cli.mode == "collect":
            if args_cli.command_checkpoint is None:
                result = _collect(env, actor, teacher_policy, checkpoint)
            else:
                result = _collect_latency(env, actor, teacher_policy, checkpoint)
        elif args_cli.mode == "dagger-collect":
            result = _collect_dagger(env, actor, teacher_policy, checkpoint)
        elif args_cli.mode == "latency-eval":
            result = _run_latency_eval(args_cli.eval_config, env, actor)
        else:
            result = _evaluate(
                env,
                actor,
                teacher_policy,
                use_oracle=args_cli.mode == "oracle-eval",
            )
            result.update(
                {
                    "mode": args_cli.mode,
                    "teacher_checkpoint": str(checkpoint),
                    "seed": args_cli.seed,
                }
            )
            if args_cli.output_dir is not None:
                args_cli.output_dir.mkdir(parents=True, exist_ok=True)
                (args_cli.output_dir / f"{args_cli.mode}.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
