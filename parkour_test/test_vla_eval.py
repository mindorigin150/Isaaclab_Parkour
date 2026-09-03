import importlib.util
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


def test_vla_eval_refreshes_reset_slots_and_truncates_final_vector_step():
    module = _load_parkour_vla_module()
    module.GO2_PARKOUR_YAW_SCALE = 1.5
    module.PARKOUR_VLA_PROPRIO_DIM = 53
    calls = []
    pool = SimpleNamespace(close=lambda: None)
    module._new_policy_pool = lambda: pool
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
