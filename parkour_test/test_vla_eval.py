import importlib.util
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_ppo_with_extractor(monkeypatch):
    """Load the PPO module without importing the Isaac Lab package graph."""
    rsl_rl = types.ModuleType("rsl_rl")
    algorithms = types.ModuleType("rsl_rl.algorithms")
    algorithms.PPO = type("PPO", (), {})
    storage = types.ModuleType("rsl_rl.storage")
    rollout_storage = types.ModuleType("rsl_rl.storage.rollout_storage")
    rollout_storage.RolloutStorage = type("RolloutStorage", (), {})
    rollout_storage.split_and_pad_trajectories = lambda *_args: None
    rsl_rl.__path__ = []
    algorithms.__path__ = []
    storage.__path__ = []
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_rl)
    monkeypatch.setitem(sys.modules, "rsl_rl.algorithms", algorithms)
    monkeypatch.setitem(sys.modules, "rsl_rl.storage", storage)
    monkeypatch.setitem(sys.modules, "rsl_rl.storage.rollout_storage", rollout_storage)

    package = types.ModuleType("ppo_test_package")
    package.__path__ = []
    actor_critic = types.ModuleType("ppo_test_package.actor_critic_with_encoder")
    actor_critic.ActorCriticRMA = type("ActorCriticRMA", (), {})
    monkeypatch.setitem(sys.modules, "ppo_test_package", package)
    monkeypatch.setitem(
        sys.modules, "ppo_test_package.actor_critic_with_encoder", actor_critic
    )

    script = Path(__file__).parents[1] / "scripts/rsl_rl/modules/ppo_with_extractor.py"
    spec = importlib.util.spec_from_file_location(
        "ppo_test_package.ppo_with_extractor", script
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.PPOWithExtractor


def _load_parkour_vla_module():
    script = Path(__file__).parents[1] / "scripts/rsl_rl/parkour_vla.py"
    spec = importlib.util.spec_from_file_location("parkour_vla_test_module", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np = np
    module.torch = torch
    module.args_cli = SimpleNamespace(eval_episodes=100, max_episode_steps=1500)
    module.PARKOUR_VLA_CONTROL_REPEAT = 5
    module.PARKOUR_VLA_LATENT_DIM = 32
    module.PARKOUR_VLA_YAW_DIM = 2
    module.PARKOUR_VLA_ACTION_DIM = 34
    return module


def test_latency_collection_preserves_full_issued_chunks_and_terminal_alignment(tmp_path, monkeypatch):
    """Long-lived collection contract: admitted H40 labels survive delayed control and resets."""
    module = _load_parkour_vla_module()
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    module.GO2_PARKOUR_YAW_SCALE = 1.0
    config = tmp_path / "latency.yaml"
    config.write_text(
        "env: {env_fps: 50, obs_fps: 10}\n"
        "latency: {method: fixed, fixed_latency_ms: 200, seed: 7}\n"
        "executor: {simulated_worker_capacity: 1}\n"
        "scheduler: {ordering_policy: latest_ready, hold_policy: hold}\n"
    )
    module.args_cli = SimpleNamespace(
        latency_config=config, output_dir=tmp_path / "raw", train_episodes=1,
        val_episodes=1, split_seed=0, keep_failed=True,
        command_checkpoint=tmp_path / "command.pt",
    )

    class Env:
        num_envs = 1
        device = "cpu"
        clip_actions = None

        def __init__(self):
            self.unwrapped = self
            self.obs = torch.zeros(1, 753)
            self.applied = []
            self.termination_manager = SimpleNamespace(get_term=lambda _: torch.tensor([False]))

        def get_observations(self):
            return self.obs, {}

        def step_no_reset(self, action):
            self.applied.append(action.item())
            self.obs = self.obs + 1
            return {"policy": self.obs}, torch.ones(1), self.obs[:, 0] == 12, torch.zeros(1, dtype=torch.bool), {}

        def reset(self, *, env_ids):
            self.obs = torch.zeros(1, 753)
            return {"policy": self.obs}, {}

    env = Env()
    episodes = []
    monkeypatch.setattr(module, "_rgb_frames", lambda _: np.zeros((1, 2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_dagger_shard",
        lambda output, index, rows, *, split, control_trace: episodes.append((split, rows, control_trace)))

    def command_policy(obs):
        return torch.arange(40).reshape(1, 40, 1).expand(len(obs), 40, 34) + obs[:, :1, None] * 1000

    def decoder(obs, *, hist_encoding, scandots_latent):
        return scandots_latent[:, :1]

    metadata = module._collect_latency(env, command_policy, decoder, tmp_path / "decoder.pt")
    assert metadata["raw_steps"] == 24
    assert {split for split, _, _ in episodes} == {"train", "val"}
    assert env.applied == ([0.0] * 10 + [10.0, 11.0]) * 2
    for _, rows, trace in episodes:
        np.testing.assert_array_equal(trace["control_source_raw_frame"], [-1] * 10 + [0, 0])
        np.testing.assert_array_equal(trace["control_chunk_index"], [-1] * 10 + [10, 11])
        np.testing.assert_array_equal(trace["control_applied_command"][:, 0], [0] * 10 + [10, 11])
        np.testing.assert_array_equal(trace["control_done"], [False] * 11 + [True])
        assert [row["issued_raw_frame"] for row in rows] == [0, 10]
        assert [row["ready_raw_frame"] for row in rows] == [10, 20]
        np.testing.assert_array_equal(rows[0]["action"][:, 0], np.arange(40))
        np.testing.assert_array_equal(rows[1]["action"][:, 0], np.arange(40) + 10000)
        np.testing.assert_array_equal(rows[0]["actor_observation"][:12, 0], np.arange(12))
        np.testing.assert_array_equal(rows[1]["actor_observation"][:2, 0], [10, 11])
        assert rows[0]["termination"].tolist() == [False] * 11 + [True] * 29
        assert rows[1]["termination"].tolist() == [False] + [True] * 39
        assert sum(row["raw_reward"] for row in rows) == 12


def test_single_admitted_transition_has_finite_surrogate_loss(monkeypatch):
    PPOWithExtractor = _load_ppo_with_extractor(monkeypatch)

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.history_encoder = torch.nn.Linear(1, 1)

        def infer_priv_latent(self, obs):
            return obs[:, :1] + self.history_encoder.weight[:, :1]

        def infer_hist_latent(self, obs):
            return obs[:, :1] + 2 * self.history_encoder.weight[:, :1]

    class Policy(torch.nn.Module):
        is_recurrent = False

        def __init__(self):
            super().__init__()
            self.actor = Actor()
            self.mean = torch.nn.Parameter(torch.tensor([[0.2]]))
            self.value = torch.nn.Parameter(torch.tensor([[0.5]]))
            self.distribution = None

        def act(self, obs, **_kwargs):
            self.distribution = torch.distributions.Normal(
                self.mean.expand(len(obs), -1), torch.ones((len(obs), 1))
            )
            return self.distribution.sample()

        def get_actions_log_prob(self, actions):
            return self.distribution.log_prob(actions).sum(-1)

        def evaluate(self, obs, **_kwargs):
            return self.value.expand(len(obs), 1)

        @property
        def action_mean(self):
            return self.distribution.mean

        @property
        def action_std(self):
            return self.distribution.stddev

        @property
        def entropy(self):
            return self.distribution.entropy().sum(-1)

    class Storage:
        def __init__(self):
            self.num_transitions_per_env = 1
            self.dones = torch.zeros((1, 1, 1), dtype=torch.bool)
            self.rewards = torch.ones((1, 1, 1))
            self.values = torch.zeros((1, 1, 1))
            self.returns = torch.zeros((1, 1, 1))
            self.advantages = torch.zeros((1, 1, 1))
            self.admission = torch.ones((1, 1, 1), dtype=torch.bool)

        def mini_batch_generator(self, *_args):
            yield (
                torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
                torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
                torch.tensor([[0.1], [0.2]]),
                torch.tensor([[0.0], [0.0]]),
                torch.tensor([[3.0], [100.0]]),
                torch.tensor([[1.0], [1.0]]),
                torch.tensor([[-0.9], [-0.9]]),
                torch.tensor([[0.0], [0.0]]),
                torch.tensor([[1.0], [1.0]]),
                (None, None),
                None,
                None,
                torch.tensor([[True], [False]]),
            )

        def clear(self):
            pass

    policy = Policy()
    estimator = torch.nn.Linear(1, 1)
    algorithm = PPOWithExtractor.__new__(PPOWithExtractor)
    algorithm.policy = policy
    algorithm.estimator = estimator
    algorithm.estimator_optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
    algorithm.hist_encoder_optimizer = torch.optim.Adam(
        policy.actor.history_encoder.parameters(), lr=1e-3
    )
    algorithm.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    algorithm.storage = Storage()
    algorithm.priv_states_dim = 1
    algorithm.num_prop = 1
    algorithm.num_scan = 0
    algorithm.priv_reg_coef_schedual = [0, 0, 0, 1]
    algorithm.counter = 0
    algorithm.lam = 0.95
    algorithm._latency_discounts = torch.ones_like(
        algorithm.storage.dones, dtype=torch.float32
    )
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.symmetry = None
    algorithm.normalize_advantage_per_mini_batch = True
    algorithm.num_mini_batches = 1
    algorithm.num_learning_epochs = 1
    algorithm.is_multi_gpu = False
    algorithm.max_grad_norm = 1.0
    algorithm.desired_kl = None
    algorithm.schedule = "fixed"
    algorithm.clip_param = 0.2
    algorithm.value_loss_coef = 1.0
    algorithm.entropy_coef = 0.0
    algorithm.use_clipped_value_loss = True

    algorithm.normalize_advantage_per_mini_batch = False
    algorithm.compute_returns(torch.zeros((1, 2)))
    assert torch.isfinite(algorithm.storage.advantages).all()
    algorithm.normalize_advantage_per_mini_batch = True

    losses = algorithm.update()

    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


class _FakeEnv:
    def __init__(self, limits=(1, 2, 3)):
        self.num_envs = 3
        self.device = torch.device("cpu")
        self._elapsed = torch.zeros(self.num_envs, dtype=torch.long)
        self._limits = torch.tensor(limits)
        self._successes = torch.zeros(self.num_envs, dtype=torch.bool)
        self._edge = SimpleNamespace(
            feet_at_edge=torch.zeros((self.num_envs, 4), dtype=torch.bool)
        )
        self._parkour = SimpleNamespace(
            cur_goal_idx=torch.zeros(self.num_envs, dtype=torch.long),
            num_goals=1,
        )
        self.unwrapped = SimpleNamespace(
            reward_manager=SimpleNamespace(
                get_term_cfg=lambda name: SimpleNamespace(func=self._edge)
            ),
            parkour_manager=SimpleNamespace(get_term=lambda name: self._parkour),
            termination_manager=SimpleNamespace(
                get_term=lambda name: self._successes
            ),
        )

    def get_observations(self):
        return torch.zeros((self.num_envs, 53)), {}

    def step(self, actions):
        self._elapsed += 1
        dones = self._elapsed >= self._limits
        self._successes = dones.clone()
        self._elapsed[dones] = 0
        return (
            torch.zeros((self.num_envs, 53)),
            torch.ones(self.num_envs),
            dones.long(),
            {},
        )


def test_eval_collects_exact_episode_budget_across_resets():
    module = _load_parkour_vla_module()

    result = module._evaluate(
        _FakeEnv(),
        actor=None,
        teacher_policy=lambda obs, **kwargs: torch.zeros((len(obs), 12)),
        use_oracle=False,
    )

    assert result["episodes"] == 100
    assert result["max_episode_steps"] == 1500
    assert result["episode_length"]["mean"] <= 3
    assert result["normalized_waypoint_progress"]["mean"] == 1.0
    assert result["edge_violation"]["mean"] == 0.0


def test_bootstrap_refreshes_camera_once_per_five_control_steps(tmp_path):
    module = _load_parkour_vla_module()
    module.GO2_PARKOUR_YAW_SCALE = 1.5
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    module.PARKOUR_VLA_PROMPT = "parkour"
    module.args_cli.output_dir = tmp_path / "raw"
    module.args_cli.train_episodes = 1
    module.args_cli.val_episodes = 0
    module.args_cli.split_seed = 0
    module.args_cli.seed = 1
    module.simulation_app = SimpleNamespace(is_running=lambda: True)
    camera_calls = []
    written = []

    class Env:
        num_envs = 1
        device = torch.device("cpu")

        def __init__(self):
            self.steps = 0
            self.success = torch.zeros(1, dtype=torch.bool)
            self.unwrapped = SimpleNamespace(
                termination_manager=SimpleNamespace(
                    get_term=lambda _name: self.success
                )
            )

        def get_observations(self):
            return torch.zeros((1, 753)), {}

        def step(self, _actions):
            self.steps += 1
            done = torch.tensor([self.steps == 10])
            self.success = done
            return torch.zeros((1, 753)), torch.ones(1), done, {}

    module._rgb_frames = lambda _env: camera_calls.append(None) or np.zeros(
        (1, 1, 1, 3), dtype=np.uint8
    )
    module._write_dagger_shard = (
        lambda _output, _index, rows, *, split: written.append((split, rows))
    )
    actor = SimpleNamespace(
        infer_scandots_latent=lambda obs: torch.zeros((len(obs), 32))
    )

    result = module._collect(
        Env(),
        actor,
        lambda obs, **_kwargs: torch.zeros((len(obs), 12)),
        Path("teacher.pt"),
    )

    assert result["control_steps"] == 10
    assert len(camera_calls) == 2
    assert len(written[0][1]) == 2


def test_rgb_frames_restores_visibility_and_composites_only_own_robot_pixels():
    module = _load_parkour_vla_module()
    visible = torch.tensor(
        [
            [[[10, 10, 10], [20, 20, 20]]],
            [[[30, 30, 30], [40, 40, 40]]],
        ],
        dtype=torch.uint8,
    )
    background = torch.tensor(
        [
            [[[110, 110, 110], [120, 120, 120]]],
            [[[130, 130, 130], [140, 140, 140]]],
        ],
        dtype=torch.uint8,
    )
    instance_ids = torch.tensor([[[5, 6]], [[5, 6]]])
    attributes = []

    class Attribute:
        def __init__(self):
            self.values = []

        def Set(self, value):
            self.values.append(value)

    class Prim:
        def CreateAttribute(self, *_args):
            attribute = Attribute()
            attributes.append(attribute)
            return attribute

    stage = SimpleNamespace(
        GetPrimsWithTypeName=lambda _name: ["/__Prototype_robot", "/World/other"],
        GetPrimAtPath=lambda _path: Prim(),
    )
    camera = SimpleNamespace(
        data=SimpleNamespace(
            output={
                "rgb": visible.clone(),
                "instance_segmentation_fast": instance_ids[..., None],
            },
            info={
                "instance_segmentation_fast": {
                    "idToLabels": {
                        5: "/World/envs/env_0/Robot",
                        6: "/World/envs/env_1/Robot",
                    }
                }
            },
        ),
        _is_outdated=torch.zeros(2, dtype=torch.bool),
    )
    camera.update = lambda *_args, **_kwargs: camera.data.output.__setitem__(
        "rgb", background.clone()
    )
    render_calls = []
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene={"vla_camera": camera},
            sim=SimpleNamespace(render=lambda: render_calls.append(None)),
        )
    )
    module.omni = SimpleNamespace(
        usd=SimpleNamespace(
            get_context=lambda: SimpleNamespace(get_stage_id=lambda: 1)
        )
    )
    module.usdrt = SimpleNamespace(
        Usd=SimpleNamespace(Stage=SimpleNamespace(Attach=lambda _stage_id: stage)),
        Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(Bool=bool)),
    )

    output = module._rgb_frames(env)

    np.testing.assert_array_equal(
        output,
        np.asarray(
            [
                [[[10, 10, 10], [120, 120, 120]]],
                [[[130, 130, 130], [40, 40, 40]]],
            ],
            dtype=np.uint8,
        ),
    )
    assert len(render_calls) == 1
    assert [attribute.values for attribute in attributes] == [[False, True]]


def test_dagger_observations_keep_request_order_and_episode_noise_seeds():
    """Native observation contract; worker affinity/batching belongs to ProcessInferencePool."""
    module = _load_parkour_vla_module()
    observations = module._policy_observations(
        np.zeros((8, 1, 1, 3), dtype=np.uint8), np.zeros((8, 53), dtype=np.float32),
        [7, 2, 5, 0], list(range(100, 108)), 10,
    )
    assert [item.metadata["slot_id"] for item in observations] == [7, 2, 5, 0]
    assert [item.metadata["action_noise_seed"] for item in observations] == [107, 102, 105, 100]
    assert all(item.env_step == 10 and item.sim_time_ms == 200 for item in observations)



def test_latency_backend_uses_no_reset_step_and_preserves_edge_moments():
    module = _load_parkour_vla_module()
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    module.GO2_PARKOUR_YAW_SCALE = 1.5
    module._rgb_frames = lambda _env: np.zeros((2, 1, 1, 3), dtype=np.uint8)

    class Edge:
        feet_at_edge = torch.tensor([[True, False], [False, True]])

    class RawEnv:
        step_dt = 0.02
        device = torch.device("cpu")
        episode_length_buf = torch.zeros(2, dtype=torch.long)
        reward_manager = SimpleNamespace(
            get_term_cfg=lambda _name: SimpleNamespace(func=Edge())
        )
        parkour_manager = SimpleNamespace(
            get_term=lambda _name: SimpleNamespace(
                cur_goal_idx=torch.zeros(2, dtype=torch.long), num_goals=4
            )
        )
        termination_manager = SimpleNamespace(
            get_term=lambda _name: torch.tensor([False, True])
        )
        action_manager = SimpleNamespace(total_action_dim=12)

        def reset(self, *, seed, env_ids):
            self.episode_length_buf[env_ids] = 0
            return {"policy": torch.zeros(2, 753)}, {}

        def step_no_reset(self, _action, *, active_mask=None):
            if active_mask is None:
                active_mask = torch.ones(2, dtype=torch.bool)
            self.episode_length_buf[active_mask] += 1
            return (
                {"policy": torch.zeros(2, 753)},
                torch.ones(2),
                torch.tensor([False, True]),
                torch.tensor([False, False]),
                {},
            )

    raw = RawEnv()
    env = SimpleNamespace(
        num_envs=2,
        unwrapped=raw,
        device=torch.device("cpu"),
        get_observations=lambda: (torch.zeros(2, 753), {}),
        close=lambda: None,
    )
    actor = lambda obs, **_kwargs: torch.zeros((len(obs), 12))
    backend = module.ParkourEnvStepBackend(
        env,
        actor,
        noop_action=module.Action(value=np.zeros(34, dtype=np.float32)),
    )

    backend.reset_slot(0, episode_id=0, seed=2)
    response = backend.step_slots(
        {0: module.Action(value=np.zeros(34, dtype=np.float32))}
    )[0].result

    assert response.done is False
    assert raw.episode_length_buf.tolist() == [1, 0]
    assert response.info["task_metric_moments"]["edge_violation"] == {
        "sum": 1.0,
        "sum_sq": 1.0,
        "count": 1,
    }


def test_latency_backend_leaves_completed_slots_inactive():
    module = _load_parkour_vla_module()
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    module.GO2_PARKOUR_YAW_SCALE = 1.5
    module._rgb_frames = lambda _env: np.zeros((2, 1, 1, 3), dtype=np.uint8)

    class RawEnv:
        step_dt = 0.02
        device = torch.device("cpu")
        episode_length_buf = torch.zeros(2, dtype=torch.long)
        reward_manager = SimpleNamespace(
            get_term_cfg=lambda _name: SimpleNamespace(
                func=SimpleNamespace(feet_at_edge=torch.zeros((2, 2), dtype=torch.bool))
            )
        )
        parkour_manager = SimpleNamespace(
            get_term=lambda _name: SimpleNamespace(
                cur_goal_idx=torch.zeros(2, dtype=torch.long), num_goals=4
            )
        )
        termination_manager = SimpleNamespace(
            get_term=lambda _name: torch.tensor([False, False])
        )
        seen_actions = []

        def step_no_reset(self, _action, *, active_mask):
            self.seen_actions.append(_action.clone())
            self.episode_length_buf[active_mask] += 1
            return (
                {"policy": torch.zeros(2, 753)},
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

    raw = RawEnv()
    env = SimpleNamespace(
        num_envs=2,
        unwrapped=raw,
        device=torch.device("cpu"),
        get_observations=lambda: (torch.zeros(2, 753), {}),
        close=lambda: None,
    )
    actor_calls = []

    def actor(obs, **_kwargs):
        actor_calls.append(None)
        return torch.full((len(obs), 12), float(len(actor_calls)))

    backend = module.ParkourEnvStepBackend(
        env,
        actor,
        noop_action=module.Action(value=np.zeros(34, dtype=np.float32)),
    )

    action = module.Action(value=np.zeros(34, dtype=np.float32))
    backend.step_slots({0: action, 1: action})
    backend.step_slots({1: action})

    assert raw.episode_length_buf.tolist() == [1, 2]
    assert raw.seen_actions[0][0, 0] == 1
    assert raw.seen_actions[1][0, 0] == 1
    assert raw.seen_actions[1][1, 0] == 2


@pytest.mark.parametrize("horizon", [1, 40])
def test_dagger_chunks_are_consumed_in_order_and_shards_do_not_mix_slots(tmp_path, horizon):
    module = _load_parkour_vla_module()
    module.args_cli.output_dir = tmp_path / "dagger"
    module.args_cli.dagger_row_budget = 4
    module.args_cli.dagger_shard_rows = 1000
    module.args_cli.dagger_round = 0
    module.args_cli.inference_batch_size = 8
    module.args_cli.seed = 1
    module.GO2_PARKOUR_YAW_SCALE = 1.5
    module.GO2_PARKOUR_MTS_THRESHOLD_RAD = 0.6
    module.PARKOUR_VLA_ACTION_DIM = 34
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    module.PARKOUR_VLA_PROMPT = "parkour"
    module.simulation_app = SimpleNamespace(is_running=lambda: True)
    module._rgb_frames = lambda env: np.zeros((env.num_envs, 1, 1, 3))
    module.apply_parkour_mts = lambda obs, yaw: (
        obs,
        torch.ones(len(obs), dtype=torch.bool),
    )
    written = []
    module._write_dagger_shard = (
        lambda output, index, rows, **kwargs: written.append(rows)
    )

    module.args_cli.action_horizon = horizon

    def predict(observations):
        chunk = np.zeros((len(observations), horizon, 34), dtype=np.float32)
        chunk[:, :, 0] = np.arange(horizon)
        return [SimpleNamespace(action_chunk=action) for action in chunk]

    module._new_policy_pool = lambda: SimpleNamespace(predict_batch=predict, close=lambda: None)
    executed = []

    class Env:
        num_envs = 2
        device = torch.device("cpu")

        def __init__(self):
            self.elapsed = torch.zeros(2, dtype=torch.long)
            self.obs = torch.zeros((2, 753))

        def observations(self):
            self.obs[:, 0] = torch.arange(2)
            self.obs[:, 1] = self.elapsed
            return self.obs

        def get_observations(self):
            return self.observations(), {}

        def step(self, actions):
            self.elapsed += 1
            dones = self.elapsed == torch.tensor([6, 100])
            self.elapsed[dones] = 0
            return self.observations(), torch.zeros(2), dones.long(), {}

    def policy(obs, *, scandots_latent, **kwargs):
        executed.append(float(scandots_latent[0, 0]))
        return torch.zeros((len(obs), 12))

    actor = SimpleNamespace(
        infer_scandots_latent=lambda obs: torch.zeros((len(obs), 32))
    )
    result = module._collect_dagger(Env(), actor, policy, Path("teacher.pt"))

    assert result["schema_version"] == 6
    assert result["rows"] == 4
    assert executed[:6] == ([0.0] * 6 if horizon == 1 else [0.0, 1.0, 2.0, 3.0, 4.0, 0.0])
    assert sorted(len(rows) for rows in written) == [2, 2]
    for rows in written:
        assert len({float(row["observation.state"][0]) for row in rows}) == 1
        assert all(row["actor_observation"].shape == (5, 753) for row in rows)
        assert all(row["termination"].shape == (5,) for row in rows)
        assert rows[-1]["termination"][-1]
        np.testing.assert_array_equal(rows[0]["actor_observation"][:, 1], [0, 1, 2, 3, 4])
