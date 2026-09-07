# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher
from tqdm import tqdm
from collections import deque
import numpy as np 
import statistics

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--eval_episodes", type=int, default=100, help="Number of complete episodes to report.")
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=1500,
    help="Maximum raw simulator steps per episode; latency control_repeat derives the policy-call limit.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--latency-config", type=Path)
parser.add_argument("--latency-teacher-checkpoint", type=Path)
parser.add_argument("--teacher-baseline", action="store_true", help="Evaluate the native teacher through delayed commands.")
parser.add_argument("--summary-path", type=Path, help="Write evaluation metrics and run identity as JSON.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
from scripts.rsl_rl.latency_vecenv import ParkourLatencyRslRlVecEnvWrapper
from parkour_isaaclab.actor import GO2_PARKOUR_YAW_SCALE

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


class TeacherCommandPolicy:
    """Expose a native teacher's latent+yaw command to the latency transport."""

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, observations, hist_encoding=True):
        del hist_encoding
        actor = self.actor
        if actor.if_scan_encode:
            scan = observations[:, actor.num_prop : actor.num_prop + actor.num_scan]
            scan_latent = actor.scan_encoder(scan)
        else:
            scan_latent = observations[:, actor.num_prop : actor.num_prop + actor.num_scan]
        return torch.cat((scan_latent, observations[:, 6:8] / GO2_PARKOUR_YAW_SCALE), dim=-1)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    if args_cli.task.find('Eval') == -1:
        print(f"[INFO] task argument must have 'Eval'")
        return 
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    teacher_baseline = False
    latency_config = None

    # wrap around environment for rsl-rl
    if args_cli.latency_config is not None:
        from latency_bench.core.config import load_config

        latency_config = load_config(args_cli.latency_config)
        command_dim = latency_config["command"]["dimension"]
        teacher_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        teacher_runner = OnPolicyRunnerWithExtractor(
            teacher_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        teacher_runner.load(str(args_cli.latency_teacher_checkpoint), load_optimizer=False)

        env = ParkourLatencyRslRlVecEnvWrapper(
            env,
            teacher_runner.alg.policy.actor,
            latency_config,
            gamma=agent_cfg.algorithm.gamma,
            control_repeat=latency_config["command"]["control_repeat"],
            clip_actions=agent_cfg.clip_actions,
        )
        teacher_baseline = args_cli.teacher_baseline
        if teacher_baseline:
            policy = TeacherCommandPolicy(teacher_runner.alg.policy.actor)
        else:
            agent_cfg.policy.actor.class_name = "CommandActor"
            agent_cfg.policy.actor.action_horizon = latency_config["command"]["horizon"]
            agent_cfg.policy.actor.command_dim = command_dim
    else:
        env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if not teacher_baseline:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path, load_optimizer=False, load_latency_state=False)
        print(ppo_runner)
        # obtain the trained policy for inference
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
            policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
            depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        else:
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # simulate environment
    total_steps = 1000
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_time_from_start = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    latency_admitted = 0
    latency_dropped = 0
    latency_raw_frames = 0
    total_raw_frames = 0
    latency_samples_ms = []

    reward_feet_edge = env.unwrapped.reward_manager.get_term_cfg("reward_feet_edge").func
    base_parkour = env.unwrapped.parkour_manager.get_term("base_parkour")
    # while simulation_app.is_running():
    control_repeat = latency_config["command"]["control_repeat"] if latency_config is not None else 1
    max_policy_steps = math.ceil(args_cli.max_episode_steps / control_repeat)
    evaluation_started = time.perf_counter()
    for i in tqdm(range(max_policy_steps)):
        start_time = time.time()
        # run everything in inference mode
        if agent_cfg.algorithm.class_name != "DistillationWithExtractor":
            with torch.inference_mode():
                # agent stepping
                # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding=True)
            # env stepping
        else:
            depth_camera = extras["observations"]['depth_camera'].to(env.device)
            with torch.inference_mode():
                if env.unwrapped.common_step_counter %5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
        cur_goal_idx = base_parkour.cur_goal_idx.clone()
        if latency_config is None:
            obs, rews, dones, extras = env.step(actions)
        else:
            remaining_raw_steps = args_cli.max_episode_steps - cur_episode_length.to(dtype=torch.long)
            obs, rews, dones, extras = env.step(actions, raw_step_budget=remaining_raw_steps)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        edge_violation_buffer.extend(reward_feet_edge.feet_at_edge.sum(dim=1).float().cpu().numpy().tolist())
        if latency_config is None:
            cur_reward_sum += rews
        else:
            cur_reward_sum += extras["latency_raw_reward"]
        if latency_config is None:
            executed_steps = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
        else:
            executed_steps = extras["latency_executed_steps"]
        cur_episode_length += executed_steps
        cur_time_from_start += 1
        total_raw_frames += int(executed_steps.sum().item())

        if "latency_admission" in extras:
            admission = extras["latency_admission"]
            active = extras["latency_active"]
            latency_admitted += int(admission.sum().item())
            latency_dropped += int((~admission & active).sum().item())
            latency_raw_frames += int(executed_steps.sum().item())
            sampled_ms = extras["latency_sampled_ms"].detach().cpu()
            latency_samples_ms.extend(sampled_ms[(admission & active).cpu()].tolist())
        
        new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if new_ids.numel():
            remaining = args_cli.eval_episodes - len(rewbuffer)
            new_ids = new_ids[:remaining]
            rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
            num_waypoints_buffer.extend(cur_goal_idx[new_ids].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0
            cur_time_from_start[new_ids] = 0

        if len(rewbuffer) >= args_cli.eval_episodes:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    rew_mean = statistics.mean(rewbuffer)
    rew_std = statistics.stdev(rewbuffer)

    len_mean = statistics.mean(lenbuffer)
    len_std = statistics.stdev(lenbuffer)

    num_waypoints_mean = np.mean(np.array(num_waypoints_buffer).astype(float)/7.0)
    num_waypoints_std = np.std(np.array(num_waypoints_buffer).astype(float)/7.0)

    edge_violation_mean = np.mean(edge_violation_buffer)
    edge_violation_std = np.std(edge_violation_buffer)
    elapsed_seconds = time.perf_counter() - evaluation_started

    print("Mean reward: {:.2f}$\pm${:.2f}".format(rew_mean, rew_std), flush=True)
    print("Mean episode length: {:.2f}$\pm${:.2f}".format(len_mean, len_std), flush=True)
    print("Mean number of waypoints: {:.2f}$\pm${:.2f}".format(num_waypoints_mean, num_waypoints_std), flush=True)
    print("Mean edge violation: {:.2f}$\pm${:.2f}".format(edge_violation_mean, edge_violation_std), flush=True)
    if latency_config is not None:
        print(
            "Latency admissions: {} drops: {} raw_control_frames: {} sampled_ms_mean: {:.2f}".format(
                latency_admitted,
                latency_dropped,
                latency_raw_frames,
                np.mean(latency_samples_ms),
            ),
            flush=True,
        )
    if args_cli.summary_path is not None:
        latency_identity = None
        if latency_config is not None:
            latency_identity = {
                "method": latency_config["latency"]["method"],
                "env_fps": latency_config["env"]["env_fps"],
                "obs_fps": latency_config["env"]["obs_fps"],
                "profile_path": (
                    latency_config["latency"]["profile_path"]
                    if latency_config["latency"]["method"] == "temporal"
                    else None
                ),
            }
        summary = {
            "identity": {
                "checkpoint": str(resume_path),
                "latency_config": str(args_cli.latency_config) if args_cli.latency_config is not None else None,
                "latency": latency_identity,
                "teacher_checkpoint": (
                    str(args_cli.latency_teacher_checkpoint)
                    if args_cli.latency_teacher_checkpoint is not None
                    else None
                ),
                "teacher_baseline": teacher_baseline,
                "seed": env_cfg.seed,
            },
            "evaluation": {
                "num_envs": args_cli.num_envs,
                "completed_episodes": len(rewbuffer),
                "requested_episodes": args_cli.eval_episodes,
                "max_episode_steps": args_cli.max_episode_steps,
                "control_repeat": control_repeat,
            },
            "metrics": {
                "reward_mean": rew_mean,
                "reward_std": rew_std,
                "episode_length_raw_mean": len_mean,
                "episode_length_raw_std": len_std,
                "waypoints_mean": num_waypoints_mean,
                "waypoints_std": num_waypoints_std,
                "edge_violation_mean": edge_violation_mean,
                "edge_violation_std": edge_violation_std,
            },
            "latency": {
                "scope": "all_vector_slots_until_requested_episode_count",
                "admissions": latency_admitted,
                "drops": latency_dropped,
                "raw_control_frames": latency_raw_frames,
                "sampled_ms_mean": np.mean(latency_samples_ms) if latency_samples_ms else None,
            },
            "throughput": {
                "raw_control_frames": total_raw_frames,
                "elapsed_seconds": elapsed_seconds,
                "raw_control_frames_per_second": total_raw_frames / elapsed_seconds,
            },
        }
        args_cli.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args_cli.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
