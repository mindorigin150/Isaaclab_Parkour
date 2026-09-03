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
import subprocess
import sys
from pathlib import Path

PARKOUR_TASK = "Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0"
PARKOUR_EVAL_MODES = {"teacher-eval", "oracle-eval", "vla-eval"}
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
            "vla-eval",
            "dagger-collect",
        ),
    )
    parser.add_argument("--task", default=PARKOUR_TASK)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=1500)
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
    parser.add_argument("--dagger_round", type=int, default=0)
    parser.add_argument("--dagger_row_budget", type=int, default=64_000)
    parser.add_argument("--dagger_shard_rows", type=int, default=1_000)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.num_envs is None:
        args_cli.num_envs = (
            192
            if args_cli.mode == "dagger-collect"
            else 50
        )
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import omni.usd
    import torch
    import usdrt
    from PIL import Image

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

    runner = OnPolicyRunnerWithExtractor(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return env, runner.alg.policy.actor, policy, Path(resume_path)


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


def _masked_vla_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).copy()
    state[:, 6:8] = 0.0
    return state


def _predict_vla_outputs(
    pool, rgb: np.ndarray, state: np.ndarray, slots: list[int], step: int
) -> np.ndarray:
    state = _masked_vla_state(state)
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


def _actor_observation_with_yaw(obs, normalized_yaw):
    actor_obs = obs.clone()
    actor_obs[:, 6:8] = normalized_yaw * GO2_PARKOUR_YAW_SCALE
    return actor_obs


def _evaluate(env, actor, teacher_policy, *, use_oracle: bool, use_vla: bool) -> dict:
    pool = _new_policy_pool() if use_vla else None
    obs, _ = env.get_observations()
    num_envs = env.num_envs
    phase = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    latent = torch.zeros(
        (num_envs, PARKOUR_VLA_LATENT_DIM), dtype=obs.dtype, device=env.device
    )
    predicted_yaw = torch.zeros(
        (num_envs, PARKOUR_VLA_YAW_DIM), dtype=obs.dtype, device=env.device
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

    try:
        step = 0
        while len(episode_returns) < args_cli.eval_episodes:
            due_ids = (phase == 0).nonzero(as_tuple=False).flatten()
            with torch.inference_mode():
                if use_oracle and due_ids.numel():
                    latent[due_ids] = actor.infer_scandots_latent(obs[due_ids])
                elif use_vla and due_ids.numel():
                    slots = due_ids.cpu().tolist()
                    predicted = _predict_vla_outputs(
                        pool,
                        _rgb_frames(env),
                        obs[:, :PARKOUR_VLA_PROPRIO_DIM].detach().cpu().numpy(),
                        slots,
                        step,
                    )
                    latent[due_ids] = torch.from_numpy(
                        predicted[:, :PARKOUR_VLA_LATENT_DIM]
                    ).to(env.device)
                    predicted_yaw[due_ids] = torch.from_numpy(
                        predicted[:, PARKOUR_VLA_LATENT_DIM :]
                    ).to(env.device)

                actor_obs = (
                    _actor_observation_with_yaw(obs, predicted_yaw)
                    if use_vla
                    else obs
                )
                actions = (
                    teacher_policy(actor_obs, hist_encoding=True, scandots_latent=latent)
                    if use_oracle or use_vla
                    else teacher_policy(actor_obs, hist_encoding=True)
                )

            goal_index = parkour.cur_goal_idx.clone()
            obs, rewards, dones, _ = env.step(actions)
            successes = env.unwrapped.termination_manager.get_term(
                "parkour_success"
            )
            edge = edge_term.feet_at_edge.sum(dim=1).float()
            for slot, value in enumerate(edge.cpu().numpy().tolist()):
                episode_edges[slot].append(value)
            returns += rewards
            lengths += 1
            phase = (phase + 1) % PARKOUR_VLA_CONTROL_REPEAT

            done_ids = dones.bool().nonzero(as_tuple=False).flatten()
            if done_ids.numel():
                remaining = args_cli.eval_episodes - len(episode_returns)
                accepted_ids = done_ids[:remaining]
                episode_returns.extend(returns[accepted_ids].cpu().numpy().tolist())
                episode_lengths.extend(lengths[accepted_ids].cpu().numpy().tolist())
                episode_progress = (
                    goal_index[accepted_ids] + successes[accepted_ids].long()
                ).float() / parkour.num_goals
                progress.extend(episode_progress.cpu().numpy().tolist())
                for slot in accepted_ids.cpu().tolist():
                    edge_violations.extend(episode_edges[slot])
                returns[done_ids] = 0
                lengths[done_ids] = 0
                phase[done_ids] = 0
                for slot in done_ids.cpu().tolist():
                    episode_edges[slot].clear()
            step += 1
    finally:
        if pool is not None:
            pool.close()

    return {
        "episodes": len(episode_returns),
        "max_episode_steps": args_cli.max_episode_steps,
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
                    "action": torch.cat(
                        (latent[slot], obs[slot, 6:8] / 1.5), dim=0
                    ).detach().cpu().numpy().tolist(),
                    "action_text": "terrain_latent_yaw",
                    "state": _masked_vla_state(
                        obs[slot, :PARKOUR_VLA_PROPRIO_DIM]
                        .detach()
                        .cpu()
                        .numpy()[None]
                    )[0].tolist(),
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
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _write_dagger_shard(
    output_dir: Path, shard_index: int, rows: list[dict]
) -> tuple[Path, Path]:
    video_tmp = output_dir / f".dagger_shard_{shard_index:06d}.tmp.mp4"

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{rows[0]['rgb'].shape[1]}x{rows[0]['rgb'].shape[0]}",
            "-framerate",
            "10",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_tmp),
        ],
        stdin=subprocess.PIPE,
    )
    for row in rows:
        process.stdin.write(np.asarray(row["rgb"], dtype=np.uint8).tobytes())
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)

    payload = {
        "state": np.stack(
            [row["observation.state"] for row in rows]
        ).astype(np.float32),
        "action": np.stack([row["action"] for row in rows]).astype(np.float32),
        "termination": np.stack([row["termination"] for row in rows]).astype(bool),
        "image_shape": np.asarray(rows[0]["rgb"].shape, dtype=np.int64),
    }
    # Only DAgger collection depends on latency_bench; other Parkour modes stay standalone.
    from latency_bench.data.parkour_dagger import write_parkour_dagger_shard

    shard_dir = write_parkour_dagger_shard(
        output_dir,
        split="train",
        episode_idx=shard_index,
        arrays=payload,
        video_path=video_tmp,
    )
    video_tmp.unlink()
    return shard_dir / "episode.npz", shard_dir / "episode.mp4"


def _existing_dagger_rows(output_dir: Path) -> tuple[int, int]:
    shard_dir = output_dir / "rollout_shards" / "train"
    row_count = 0
    next_shard_id = 0
    for path in sorted(shard_dir.glob("episode_*")):
        if not path.is_dir():
            continue
        shard_path = path / "episode.npz"
        with np.load(shard_path) as shard:
            row_count += int(shard["state"].shape[0])
        next_shard_id = max(next_shard_id, int(path.name.split("_")[-1]) + 1)
    return row_count, next_shard_id


def _collect_dagger(env, actor, teacher_policy, checkpoint: Path) -> dict:
    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count, shard_index = _existing_dagger_rows(output_dir)
    target_rows = args_cli.dagger_row_budget
    if row_count >= target_rows:
        return {
            "schema_version": 4,
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

    pool = _new_policy_pool()
    obs, _ = env.get_observations()
    num_envs = env.num_envs
    all_ids = torch.arange(num_envs, device=env.device)
    latent = torch.zeros(
        (num_envs, PARKOUR_VLA_LATENT_DIM), dtype=obs.dtype, device=env.device
    )
    predicted_yaw = torch.zeros(
        (num_envs, PARKOUR_VLA_YAW_DIM), dtype=obs.dtype, device=env.device
    )
    oracle_latent = torch.zeros_like(latent)
    rows: list[dict] = []
    control_step = 0
    mts_prediction_count = 0
    mts_total_count = 0

    try:
        while row_count + len(rows) < target_rows and simulation_app.is_running():
            rgb = _rgb_frames(env)
            with torch.inference_mode():
                slots = all_ids.cpu().tolist()
                predicted = _predict_vla_outputs(
                    pool,
                    rgb,
                    obs[:, :PARKOUR_VLA_PROPRIO_DIM].detach().cpu().numpy(),
                    slots,
                    control_step,
                )
                latent[:] = torch.from_numpy(
                    predicted[:, :PARKOUR_VLA_LATENT_DIM]
                ).to(env.device)
                predicted_yaw[:] = torch.from_numpy(
                    predicted[:, PARKOUR_VLA_LATENT_DIM :]
                ).to(env.device)
                oracle_latent[:] = actor.infer_scandots_latent(obs)
            remaining = target_rows - row_count - len(rows)
            selected_ids = all_ids[:remaining]
            selected = selected_ids.cpu().tolist()

            if selected:
                selected_states = _masked_vla_state(
                    obs[selected_ids, :PARKOUR_VLA_PROPRIO_DIM]
                    .detach()
                    .cpu()
                    .numpy()
                )
                selected_actions = torch.cat(
                    (
                        oracle_latent[selected_ids],
                        obs[selected_ids, 6:8] / GO2_PARKOUR_YAW_SCALE,
                    ),
                    dim=1,
                ).detach().cpu().numpy()
                row_data = {
                    slot: {
                        "rgb": rgb[slot],
                        "observation.state": state,
                        "action": action,
                        "termination": [],
                    }
                    for slot, state, action in zip(
                        selected, selected_states, selected_actions
                    )
                }
            else:
                row_data = {}

            for repeat_index in range(PARKOUR_VLA_CONTROL_REPEAT):
                with torch.inference_mode():
                    actor_observation, use_prediction = apply_parkour_mts(
                        obs,
                        predicted_yaw,
                    )
                    mts_prediction_count += use_prediction.sum().item()
                    mts_total_count += use_prediction.numel()
                    actions = teacher_policy(
                        actor_observation,
                        hist_encoding=True,
                        scandots_latent=latent,
                    )
                obs, _, dones, _ = env.step(actions)

                if selected:
                    selected_terminations = dones[selected_ids].detach().cpu().tolist()
                    for slot, termination in zip(
                        selected,
                        selected_terminations,
                    ):
                        row_data[slot]["termination"].append(termination)

                control_step += 1
                reset_ids = dones.nonzero(as_tuple=False).flatten()
                if reset_ids.numel() and repeat_index + 1 < PARKOUR_VLA_CONTROL_REPEAT:
                    slots = reset_ids.cpu().tolist()
                    predicted = _predict_vla_outputs(
                        pool,
                        _rgb_frames(env),
                        obs[:, :PARKOUR_VLA_PROPRIO_DIM].detach().cpu().numpy(),
                        slots,
                        control_step,
                    )
                    latent[reset_ids] = torch.from_numpy(
                        predicted[:, :PARKOUR_VLA_LATENT_DIM]
                    ).to(env.device)
                    predicted_yaw[reset_ids] = torch.from_numpy(
                        predicted[:, PARKOUR_VLA_LATENT_DIM :]
                    ).to(env.device)
            for slot in selected:
                row = row_data[slot]
                row["termination"] = np.asarray(row["termination"], dtype=bool)
                rows.append(row)

            if len(rows) >= args_cli.dagger_shard_rows:
                count = args_cli.dagger_shard_rows
                _write_dagger_shard(
                    output_dir,
                    shard_index,
                    rows[:count],
                )
                row_count += count
                rows = rows[count:]
                shard_index += 1

    finally:
        if rows:
            _write_dagger_shard(
                output_dir,
                shard_index,
                rows,
            )
            row_count += len(rows)
        pool.close()

    metadata = {
        "schema_version": 4,
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
        "base_prompt": PARKOUR_VLA_PROMPT,
        "teacher_checkpoint": str(checkpoint),
        "failure_and_timeout_rows_retained": True,
        "state_yaw_indices_masked": [6, 7],
        "dagger_mts_threshold_rad": GO2_PARKOUR_MTS_THRESHOLD_RAD,
        "dagger_mts_prediction_fraction": mts_prediction_count / mts_total_count,
    }
    temporary = output_dir / "metadata.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_dir / "metadata.json")
    return metadata


def main() -> None:
    env, actor, teacher_policy, checkpoint = _load_environment_and_teacher()
    try:
        if args_cli.mode == "collect":
            result = _collect(env, actor, teacher_policy, checkpoint)
        elif args_cli.mode == "dagger-collect":
            result = _collect_dagger(env, actor, teacher_policy, checkpoint)
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
