import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


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
    return module


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
        use_vla=False,
    )

    assert result["episodes"] == 100
    assert result["max_episode_steps"] == 1500
    assert result["episode_length"]["mean"] <= 3
    assert result["normalized_waypoint_progress"]["mean"] == 1.0
    assert result["edge_violation"]["mean"] == 0.0


def test_profile_env_maps_latent_and_yaw_through_fixed_actor():
    module = _load_parkour_vla_module()
    module.PARKOUR_VLA_LATENT_DIM = 32
    module.PARKOUR_VLA_YAW_DIM = 2
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    module.GO2_PARKOUR_YAW_SCALE = 1.5
    seen = []

    class Env:
        device = torch.device("cpu")

        def __init__(self):
            self.step_count = 0
            self.edge = SimpleNamespace(
                feet_at_edge=torch.zeros((1, 4), dtype=torch.bool)
            )
            self.parkour = SimpleNamespace(
                cur_goal_idx=torch.tensor([1]), num_goals=4
            )
            self.success = torch.tensor([False])
            self.unwrapped = SimpleNamespace(
                reward_manager=SimpleNamespace(
                    get_term_cfg=lambda _name: SimpleNamespace(func=self.edge)
                ),
                parkour_manager=SimpleNamespace(
                    get_term=lambda _name: self.parkour
                ),
                termination_manager=SimpleNamespace(
                    get_term=lambda _name: self.success
                ),
            )

        def reset(self):
            self.step_count = 0
            self.parkour.cur_goal_idx = torch.tensor([1])
            self.success = torch.tensor([False])
            self.edge.feet_at_edge = torch.zeros((1, 4), dtype=torch.bool)
            return torch.zeros(1, 53), {}

        def step(self, action):
            seen.append(action)
            self.step_count += 1
            self.edge.feet_at_edge = torch.tensor(
                [[True, True, self.step_count > 1, self.step_count > 1]]
            )
            self.success = torch.tensor([self.step_count == 1])
            return (
                torch.zeros(1, 53),
                torch.tensor([2.0]),
                torch.tensor([self.step_count == 1]),
                {},
            )

    class Actor:
        def __call__(self, actor_obs, **kwargs):
            seen.append((actor_obs, kwargs))
            return torch.zeros(1, 12)

    env = Env()
    adapter = module._ParkourProfileEnv(env, Actor())
    adapter._obs = torch.zeros(1, 53)
    result = adapter.step(
        module.Action(value=np.arange(34, dtype=np.float32))
    )

    assert result.done
    assert result.reward == 2.0
    assert result.info["task_metrics"] == {
        "normalized_waypoint_progress": 0.5,
        "edge_violation": 2.0,
    }
    actor_obs, kwargs = seen[0]
    np.testing.assert_array_equal(kwargs["scandots_latent"].numpy(), np.arange(32)[None])
    np.testing.assert_array_equal(actor_obs[0, 6:8].numpy(), np.arange(32, 34) * 1.5)
    assert seen[1].shape == (1, 12)

    env.success = torch.tensor([False])
    env.edge.feet_at_edge = torch.tensor([[True, True, True, True]])
    result = adapter.step(module.Action(value=np.zeros(34, dtype=np.float32)))
    assert result.info["task_metrics"]["normalized_waypoint_progress"] == 0.25
    assert result.info["task_metrics"]["edge_violation"] == 3.0

    module._rgb_frames = lambda _env: np.zeros((1, 1, 1, 3), dtype=np.uint8)
    adapter.reset()
    assert adapter._episode_edge_sum == 0.0
    assert adapter._episode_edge_steps == 0


def test_vla_eval_refreshes_reset_slots_and_truncates_final_vector_step():
    module = _load_parkour_vla_module()
    module.GO2_PARKOUR_YAW_SCALE = 1.5
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    calls = []
    pool = SimpleNamespace(close=lambda: None)
    module._new_policy_pool = lambda: [pool]
    module._rgb_frames = lambda env: np.zeros((env.num_envs, 1, 1, 3))

    def predict(pool, rgb, state, slots, step):
        calls.append((step, slots))
        return np.zeros((len(slots), 34), dtype=np.float32)

    module._predict_vla_outputs = predict

    result = module._evaluate(
        _FakeEnv(limits=(1, 1, 1)),
        actor=None,
        teacher_policy=lambda obs, **kwargs: torch.zeros((len(obs), 12)),
        use_oracle=False,
        use_vla=True,
    )

    assert result["episodes"] == 100
    assert len(calls) == 34
    assert all(slots == [0, 1, 2] for _, slots in calls)


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


def test_multi_pool_prediction_keeps_slot_affinity_and_request_order():
    module = _load_parkour_vla_module()
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    module.args_cli.inference_batch_size = 2

    class Pool:
        def __init__(self, worker_id):
            self.worker_id = worker_id
            self.slots = []

        def predict_batch(self, observations):
            slots = [item.metadata["slot_id"] for item in observations]
            assert all(slot % 2 == self.worker_id for slot in slots)
            self.slots.extend(slots)
            if self.worker_id == 0:
                time.sleep(0.01)
            return [
                SimpleNamespace(
                    action=SimpleNamespace(
                        value=np.full(34, slot, dtype=np.float32)
                    )
                )
                for slot in slots
            ]

    pools = [Pool(0), Pool(1)]
    rgb = np.zeros((8, 1, 1, 3), dtype=np.uint8)
    state = np.zeros((8, 53), dtype=np.float32)

    output = module._predict_vla_outputs(pools, rgb, state, [7, 2, 5, 0], 10)
    reset_output = module._predict_vla_outputs(pools, rgb, state, [2, 7], 11)

    np.testing.assert_array_equal(output[:, 0], [7, 2, 5, 0])
    np.testing.assert_array_equal(reset_output[:, 0], [2, 7])
    assert pools[0].slots == [2, 0, 2]
    assert pools[1].slots == [7, 5, 7]
