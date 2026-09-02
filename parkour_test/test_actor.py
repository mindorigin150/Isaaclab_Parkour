import torch

from parkour_isaaclab.actor import (
    GO2_PARKOUR_ACTION_DIM,
    GO2_PARKOUR_ACTOR_OBSERVATION_DIM,
    GO2_PARKOUR_TERRAIN_LATENT_DIM,
    build_go2_parkour_actor,
)


def test_go2_actor_uses_external_latent_and_history():
    actor = build_go2_parkour_actor()

    action = actor(
        torch.randn(2, GO2_PARKOUR_ACTOR_OBSERVATION_DIM),
        hist_encoding=True,
        scandots_latent=torch.randn(2, GO2_PARKOUR_TERRAIN_LATENT_DIM),
    )

    assert action.shape == (2, GO2_PARKOUR_ACTION_DIM)
