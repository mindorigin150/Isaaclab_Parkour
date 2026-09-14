"""Long-lived native PPO contracts for command admission, transport, and logging."""

import math
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modules.actor_critic_with_encoder import ActorCriticRMA
from modules.ppo_with_extractor import PPOWithExtractor
from modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from training.common.action_latency import ActionLatencyBatch


def test_runner_logs_control_steps_and_undiscounted_episode_rewards(tmp_path, monkeypatch):
    """Long-lived logging contract: sample clocks and episode metrics match teacher units."""
    observation = torch.zeros(2, 1)
    steps = iter([torch.tensor([5, 2]), torch.tensor([1, 3])])
    raw_rewards = iter([torch.tensor([10., 20.]), torch.tensor([30., 40.])])
    rewards = iter([torch.tensor([1., 2.]), torch.tensor([3., 4.])])
    dones = iter([torch.tensor([0, 0]), torch.tensor([1, 1])])
    processed_rewards = []
    other_rank_steps = iter([5, 7])
    metrics = {}

    class Env:
        num_envs = 2
        device = "cpu"

        def get_observations(self):
            return observation, {"observations": {"policy": observation}}

        def step(self, actions):
            return observation, next(rewards), next(dones), {
                "observations": {"policy": observation},
                "latency_executed_steps": next(steps),
                "latency_raw_reward": next(raw_rewards),
            }

    class Writer:
        def add_scalar(self, tag, value, step):
            metrics[tag] = value

    def all_reduce(value):
        value += next(other_rank_steps)

    runner = object.__new__(OnPolicyRunnerWithExtractor)
    runner.alg = SimpleNamespace(
        policy=SimpleNamespace(action_std=torch.ones(2)), rnd=None, learning_rate=1e-4,
        act=lambda *args: torch.zeros(2, 34),
        process_env_step=lambda reward, *args: processed_rewards.append(reward.clone()),
        compute_returns=lambda *args: None,
        update=lambda: {"value_function": 0.}, update_dagger=lambda: 0.,
        broadcast_parameters=lambda: None,
    )
    runner.env = Env()
    runner.device = "cpu"
    runner.training_type = "rl"
    runner.writer = Writer()
    runner.log_dir = str(tmp_path)
    runner.logger_type = "tensorboard"
    runner.disable_logs = False
    runner.is_distributed = True
    runner.gpu_world_size = 2
    runner.gpu_global_rank = 0
    runner.num_steps_per_env = 1
    runner.save_interval = 100
    runner.current_learning_iteration = 0
    runner.tot_timesteps = runner.tot_control_steps = runner.tot_time = 0
    runner.dagger_update_freq = 20
    runner.mean_hist_latent_loss = 0.
    runner.privileged_obs_type = None
    runner.obs_normalizer = torch.nn.Identity()
    runner.empirical_normalization = False
    runner.git_status_repos = []
    runner.train_mode = lambda: None
    runner.save = lambda *args: None
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(sys.modules[OnPolicyRunnerWithExtractor.__module__], "store_code_state", lambda *args: [])

    runner.learn_rl(2)

    assert metrics["Train/control_steps"] == 23
    assert metrics["Train/iteration"] == 1
    assert metrics["Train/mean_reward"] == 50.
    assert metrics["Train/mean_episode_length"] == 5.5
    torch.testing.assert_close(torch.stack(processed_rewards), torch.tensor([[1., 2.], [3., 4.]]))


def test_wandb_step_matches_existing_teacher_sampling_axis(tmp_path, monkeypatch):
    """Long-lived W&B contract: Step permits overlaying the zero-based teacher run."""
    import wandb
    from modules.teacher_step_wandb_writer import TeacherStepWandbWriter

    monkeypatch.setenv("WANDB_MODE", "disabled")
    writer = TeacherStepWandbWriter(str(tmp_path), 10, {"wandb_project": "parkour-budget-test"})
    logged = []
    monkeypatch.setattr(wandb, "log", lambda values, step: logged.append((values, step)))
    for control_steps in [1, 5 * 147456 - 1, 10 * 147456]:
        writer.control_steps = control_steps
        writer.add_scalar("Train/control_steps", control_steps, global_step=0)
    writer.close()
    wandb.finish()
    assert [step for values, step in logged] == [0, 4, 9]
    assert [values["Train/control_steps"] for values, step in logged] == [1, 737279, 1474560]


@pytest.mark.parametrize("per_minibatch", [False, True])
def test_admitted_commands_preserve_sample_alignment(per_minibatch):
    torch.manual_seed(7)
    policy = ActorCriticRMA(
        num_critic_obs=753, num_actions=34,
        actor_hidden_dims=[16], critic_hidden_dims=[16],
        scan_encoder_dims=[32], priv_encoder_dims=[20],
        tanh_encoder_output=False, noise_std_type="log",
        actor={
            "class_name": "CommandActor", "num_prop": 53, "num_scan": 132,
            "num_priv_explicit": 9, "num_priv_latent": 29, "num_hist": 10,
            "state_history_encoder": {"class_name": "StateHistoryEncoder", "channel_size": 10},
        },
    )
    algorithm = PPOWithExtractor(
        policy, torch.nn.Linear(53, 9),
        estimator_paras={
            "num_priv_explicit": 9, "num_prop": 53, "num_scan": 132,
            "learning_rate": 1e-4, "train_with_estimated_states": False,
        },
        num_learning_epochs=2, num_mini_batches=1,
        normalize_advantage_per_mini_batch=per_minibatch,
        priv_reg_coef_schedual=[0, 0, 0, 1],
    )
    algorithm.init_storage("rl", 3, 2, [753], [753], [34])
    before = policy.actor.actor_backbone[-1].weight.detach().clone()
    assert torch.count_nonzero(before) > 0
    for _ in range(2):
        for _ in range(2):
            observation = torch.randn(3, 753)
            issued = algorithm.act(observation, observation)
            assert issued.shape == (3, 34)
            algorithm.process_env_step(
                torch.tensor([1., 2., 3.]), torch.zeros(3, dtype=torch.bool),
                {"latency_discount": torch.full((3,), 0.99),
                 "latency_admission": torch.tensor([True, False, True])},
            )
        algorithm.compute_returns(torch.randn(3, 753))
        losses = algorithm.update()
        assert all(math.isfinite(value) for value in losses.values())
        assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert not torch.equal(before, policy.actor.actor_backbone[-1].weight)


def test_inference_load_does_not_restore_latency_transport_state(tmp_path):
    class Loader:
        def load_state_dict(self, state):
            self.state = state

    class Env:
        def __init__(self):
            self.loaded = False

        def load_latency_state_dict(self, state):
            self.loaded = True

    runner = object.__new__(OnPolicyRunnerWithExtractor)
    runner.alg = SimpleNamespace(policy=Loader(), estimator=Loader(), rnd=None)
    runner.depth_encoder_cfg = None
    runner.empirical_normalization = False
    runner.env = Env()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {},
            "estimator_state_dict": {},
            "latency_state_dict": {"samplers": [], "episodes": []},
            "infos": None,
        },
        checkpoint,
    )

    runner.load(str(checkpoint), load_optimizer=False, load_latency_state=False)

    assert not runner.env.loaded


@pytest.mark.parametrize(("budget", "end_steps"), [
    (None, [1, 3, 100]),
    ([5, 5, 2], [1, 3, 100]),
    ([0, 5, 2], [1, 3, 100]),
    (None, [3, 1, 2]),
    (None, [100, 100, 100]),
])
def test_latency_step_preserves_partial_budgets_and_terminal_bootstrap(budget, end_steps, monkeypatch):
    """Native adapter contract: rewards and terminal states stop at each slot's last step."""
    package = types.ModuleType("latency_adapter_test")
    package.__path__ = []
    base = types.ModuleType("latency_adapter_test.vecenv_wrapper")
    base.ParkourRslRlVecEnvWrapper = type("Base", (), {
        "unwrapped": property(lambda self: self.env),
    })
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    path = Path(__file__).parents[1] / "scripts/rsl_rl/latency_vecenv.py"
    spec = importlib.util.spec_from_file_location("latency_adapter_test.wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Env:
        cfg = SimpleNamespace(is_finite_horizon=False)

        def __init__(self):
            self.elapsed = torch.zeros(3, dtype=torch.long)

        def observations(self):
            return {"policy": self.elapsed[:, None].expand(3, 53).float().clone()}

        def step_no_reset(self, action, active_mask):
            self.elapsed += active_mask
            ended = self.elapsed == torch.tensor(end_steps)
            terminated = ended & torch.tensor([False, True, True])
            truncated = ended & torch.tensor([True, False, False])
            reward = torch.where(active_mask, self.elapsed.float(), float("nan"))
            return self.observations(), reward, terminated, truncated, {}

        def reset(self, env_ids):
            self.reset_ids = env_ids.tolist()
            self.elapsed[env_ids] = 0
            return self.observations(), {}

    class Decoder(torch.nn.Module):
        def forward(self, observation, **kwargs):
            return torch.zeros(3, 12)

    config = {
        "env": {"env_fps": 50, "obs_fps": 10},
        "latency": {"method": "fixed", "fixed_latency_ms": 0, "seed": 0},
        "executor": {"simulated_worker_capacity": 1},
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold"},
    }
    wrapper = object.__new__(module.ParkourLatencyRslRlVecEnvWrapper)
    wrapper.env = Env()
    wrapper.num_envs = 3
    wrapper.device = "cpu"
    wrapper.command_dim = 34
    wrapper.control_repeat = 5
    wrapper.gamma = 0.5
    wrapper.clip_actions = None
    wrapper.decoder = Decoder()
    wrapper._raw_obs = torch.zeros(3, 53)
    wrapper._actor_obs = torch.empty_like(wrapper._raw_obs)
    wrapper._raw_step_budget = torch.full((3,), 5, dtype=torch.long)
    wrapper._gamma = torch.full((3,), 0.5)
    wrapper._scheduler = ActionLatencyBatch(config, num_envs=3, device="cpu", noop_command=torch.zeros(34))
    wrapper._scheduler.reset()
    raw_budget = None if budget is None else torch.tensor(budget)
    observation, reward, done, info = wrapper.step(torch.ones(3, 34), raw_budget)
    budgets = [5, 5, 5] if budget is None else budget
    executed = [min(5, available, end) for available, end in zip(budgets, end_steps)]
    ended = [count == end for count, end in zip(executed, end_steps)]
    torch.testing.assert_close(info["latency_executed_steps"], torch.tensor(executed))
    torch.testing.assert_close(reward, torch.tensor([
        sum((i + 1) * 0.5**i for i in range(count)) for count in executed
    ]))
    torch.testing.assert_close(info["latency_raw_reward"], torch.tensor([
        float(sum(range(1, count + 1))) for count in executed
    ]))
    torch.testing.assert_close(info["latency_discount"], torch.tensor([0.5**count for count in executed]))
    assert done.tolist() == ended
    assert info["time_outs"].tolist() == [ended[0], False, False]
    assert info["latency_admission"].tolist() == [available > 0 for available in budgets]
    if any(ended):
        for slot, is_done in enumerate(ended):
            if is_done:
                assert info["terminal_observations"]["policy"][slot, 0] == executed[slot]
        assert wrapper.env.reset_ids == [slot for slot, is_done in enumerate(ended) if is_done]
    else:
        assert "terminal_observations" not in info
    assert observation[:, 0].tolist() == [0 if is_done else count for count, is_done in zip(executed, ended)]
    assert wrapper._scheduler.episodes == [
        3 + sum(ended[:slot]) if is_done else slot for slot, is_done in enumerate(ended)
    ]
