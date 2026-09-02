# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect and evaluate the fixed-controller Extreme Parkour VLA benchmark."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PARKOUR_TASK = "Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    import cli_args  # isort: skip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("teacher-eval", "oracle-eval", "collect", "vla-eval"),
    )
    parser.add_argument("--task", default=PARKOUR_TASK)
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--disable_fabric", action="store_true")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--train_episodes", type=int, default=250)
    parser.add_argument("--val_episodes", type=int, default=25)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--policy_config", type=Path)
    parser.add_argument("--inference_device", default="cuda:1")
    parser.add_argument("--inference_batch_size", type=int, default=8)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import isaaclab_tasks  # noqa: F401
    from parkour_tasks.extreme_parkour_task.config.go2.parkour_vla_cfg import (
        PARKOUR_VLA_LATENT_DIM,
        PARKOUR_VLA_PROMPT,
        PARKOUR_VLA_PROPRIO_DIM,
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
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return env, runner.alg.policy.actor, policy, Path(resume_path)


def _rgb_frames(env) -> np.ndarray:
    return (
        env.unwrapped.scene["vla_camera"]
        .data.output["rgb"][..., :3]
        .detach()
        .cpu()
        .numpy()
    )


def _policy_observations(rgb: np.ndarray, state: np.ndarray, slots: list[int], step: int):
    from latency_bench.core.types import Observation
    from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY

    return [
        Observation(
            data=rgb[slot],
            env_step=step,
            sim_time_ms=step * 20.0,
            metadata={
                ENV_RAW_RGB_FRAME_STACK_INFO_KEY: rgb[slot][None],
                "parkour_proprio": state[slot],
                "slot_id": slot,
            },
        )
        for slot in slots
    ]


def _predict_latents(pool, rgb: np.ndarray, state: np.ndarray, slots: list[int], step: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(slots), args_cli.inference_batch_size):
        batch_slots = slots[start : start + args_cli.inference_batch_size]
        outputs = pool.predict_batch(
            _policy_observations(rgb, state, batch_slots, step)
        )
        chunks.extend(np.asarray(output.action.value, dtype=np.float32) for output in outputs)
    return np.stack(chunks)


def _new_policy_pool():
    from latency_bench.core.config import load_config
    from latency_bench.executors.realtime.pool import ProcessInferencePool

    config = load_config(args_cli.policy_config)
    print("[INFO]: Starting the official GR00T inference worker.", flush=True)
    pool = ProcessInferencePool(
        config=config,
        inference_devices=[args_cli.inference_device],
    )
    print("[INFO]: Official GR00T inference worker is ready.", flush=True)
    return pool


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def _evaluate(env, actor, teacher_policy, *, use_oracle: bool, use_vla: bool) -> dict:
    pool = _new_policy_pool() if use_vla else None
    obs, _ = env.get_observations()
    num_envs = env.num_envs
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    phase = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    latent = torch.zeros(
        (num_envs, PARKOUR_VLA_LATENT_DIM), dtype=obs.dtype, device=env.device
    )
    returns = torch.zeros(num_envs, dtype=torch.float, device=env.device)
    lengths = torch.zeros(num_envs, dtype=torch.float, device=env.device)
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    progress: list[float] = []
    edge_violations: list[float] = []
    edge_term = env.unwrapped.reward_manager.get_term_cfg("reward_feet_edge").func
    parkour = env.unwrapped.parkour_manager.get_term("base_parkour")

    try:
        for step in range(args_cli.max_steps):
            due = active & (phase == 0)
            due_ids = due.nonzero(as_tuple=False).flatten()
            with torch.inference_mode():
                if use_oracle and due_ids.numel():
                    latent[due_ids] = actor.infer_scandots_latent(obs[due_ids])
                elif use_vla and due_ids.numel():
                    slots = due_ids.cpu().tolist()
                    predicted = _predict_latents(
                        pool,
                        _rgb_frames(env),
                        obs[:, :PARKOUR_VLA_PROPRIO_DIM].detach().cpu().numpy(),
                        slots,
                        step,
                    )
                    latent[due_ids] = torch.from_numpy(predicted).to(env.device)

                actions = (
                    teacher_policy(obs, hist_encoding=True, scandots_latent=latent)
                    if use_oracle or use_vla
                    else teacher_policy(obs, hist_encoding=True)
                )

            current_active = active.clone()
            goal_index = parkour.cur_goal_idx.clone()
            obs, rewards, dones, _ = env.step(actions)
            successes = env.unwrapped.termination_manager.get_term(
                "parkour_success"
            )
            edge = edge_term.feet_at_edge.sum(dim=1).float()
            edge_violations.extend(edge[current_active].cpu().numpy().tolist())
            returns[current_active] += rewards[current_active]
            lengths[current_active] += 1
            phase[current_active] = (phase[current_active] + 1) % 5

            done_ids = (current_active & dones.bool()).nonzero(as_tuple=False).flatten()
            if done_ids.numel():
                episode_returns.extend(returns[done_ids].cpu().numpy().tolist())
                episode_lengths.extend(lengths[done_ids].cpu().numpy().tolist())
                episode_progress = (
                    goal_index[done_ids] + successes[done_ids].long()
                ).float() / parkour.num_goals
                progress.extend(episode_progress.cpu().numpy().tolist())
                active[done_ids] = False
            if not active.any():
                break
    finally:
        if pool is not None:
            pool.close()

    return {
        "episodes": len(episode_returns),
        "reward": _summary(episode_returns),
        "episode_length": _summary(episode_lengths),
        "normalized_waypoint_progress": _summary(progress),
        "edge_violation": _summary(edge_violations),
    }


def _candidate_dir(output_dir: Path, slot: int, generation: int) -> Path:
    return output_dir / "images" / "candidates" / f"slot_{slot:03d}_{generation:06d}"


def _write_episode_shard(output_dir: Path, rows: list[dict]) -> None:
    path = (
        output_dir
        / "episode_rows"
        / rows[0]["split"]
        / f"episode_{rows[0]['episode_idx']:06d}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    temporary.replace(path)


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
    generations = [0] * num_envs
    rows_by_slot: list[list[dict]] = [[] for _ in range(num_envs)]
    parkour = env.unwrapped.parkour_manager.get_term("base_parkour")
    accepted = 0
    control_step = 0

    while accepted < episode_target and simulation_app.is_running():
        due_ids = (phase == 0).nonzero(as_tuple=False).flatten()
        rgb = _rgb_frames(env)
        with torch.inference_mode():
            latent[due_ids] = actor.infer_scandots_latent(obs[due_ids])

        for slot in due_ids.cpu().tolist():
            candidate_dir = _candidate_dir(output_dir, slot, generations[slot])
            candidate_dir.mkdir(parents=True, exist_ok=True)
            image_path = candidate_dir / f"frame_{len(rows_by_slot[slot]):06d}.png"
            Image.fromarray(rgb[slot]).save(image_path)
            rows_by_slot[slot].append(
                {
                    "image": str(image_path),
                    "env_name": "extreme_parkour_go2",
                    "episode_idx": -1,
                    "decision_step": len(rows_by_slot[slot]),
                    "split": "",
                    "seed": int(args_cli.seed),
                    "prompt": PARKOUR_VLA_PROMPT,
                    "latency_raw_frames": 0,
                    "latency_ms": 0.0,
                    "raw_reward": 0.0,
                    "action": latent[slot].detach().cpu().numpy().tolist(),
                    "action_text": "terrain_latent",
                    "state": obs[slot, :PARKOUR_VLA_PROPRIO_DIM]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                }
            )

        with torch.inference_mode():
            actions = teacher_policy(obs, hist_encoding=True, scandots_latent=latent)
        obs, rewards, dones, _ = env.step(actions)
        successes = env.unwrapped.termination_manager.get_term("parkour_success")
        for slot in range(num_envs):
            rows_by_slot[slot][-1]["raw_reward"] += float(rewards[slot].item())
        phase = (phase + 1) % 5
        control_step += 1

        for slot in dones.nonzero(as_tuple=False).flatten().cpu().tolist():
            success = bool(successes[slot].item())
            candidate_dir = _candidate_dir(output_dir, slot, generations[slot])
            if success and accepted < episode_target:
                split = split_order[accepted]
                for row in rows_by_slot[slot]:
                    row["episode_idx"] = accepted
                    row["split"] = split
                _write_episode_shard(output_dir, rows_by_slot[slot])
                accepted += 1
                print(f"accepted {accepted}/{episode_target}: split={split} slot={slot}")
            else:
                shutil.rmtree(candidate_dir)
            rows_by_slot[slot] = []
            generations[slot] += 1
            phase[slot] = 0

    for slot in range(num_envs):
        pending = _candidate_dir(output_dir, slot, generations[slot])
        if pending.exists():
            shutil.rmtree(pending)

    metadata = {
        "schema_version": 2,
        "env_name": "extreme_parkour_go2",
        "integration_name": "isaaclab_parkour",
        "action_layout": "go2_parkour_terrain_latent_v1",
        "action_labels": [
            f"terrain_latent_{index}" for index in range(PARKOUR_VLA_LATENT_DIM)
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
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    env, actor, teacher_policy, checkpoint = _load_environment_and_teacher()
    try:
        if args_cli.mode == "collect":
            result = _collect(env, actor, teacher_policy, checkpoint)
        else:
            result = _evaluate(
                env,
                actor,
                teacher_policy,
                use_oracle=args_cli.mode == "oracle-eval",
                use_vla=args_cli.mode == "vla-eval",
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
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
