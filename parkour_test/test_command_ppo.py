"""Long-lived native PPO contract: admitted full chunks retain valid minibatch updates."""

import math

import pytest
import torch

from modules.actor_critic_with_encoder import ActorCriticRMA
from modules.ppo_with_extractor import PPOWithExtractor


@pytest.mark.parametrize("per_minibatch", [False, True])
def test_admitted_chunk_updates_preserve_sample_alignment(per_minibatch):
    torch.manual_seed(7)
    policy = ActorCriticRMA(
        num_critic_obs=753, num_actions=40 * 34,
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
    algorithm.init_storage("rl", 3, 2, [753], [753], [40 * 34])
    before = policy.actor.actor_backbone[-1].weight.detach().clone()
    for _ in range(2):
        for _ in range(2):
            observation = torch.randn(3, 753)
            issued = algorithm.act(observation, observation)
            assert issued.shape == (3, 40 * 34)
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
