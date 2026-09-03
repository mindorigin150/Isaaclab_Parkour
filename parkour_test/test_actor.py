import pytest
import torch

from parkour_isaaclab.actor import (
    GO2_PARKOUR_MTS_THRESHOLD_RAD,
    GO2_PARKOUR_YAW_SCALE,
    apply_parkour_mts,
)


@pytest.mark.parametrize(
    ("error", "expected_prediction"),
    ((0.0, True), (0.59, True), (0.6, False), (1.0, False)),
)
def test_parkour_mts_uses_paper_threshold(error, expected_prediction):
    observation = torch.zeros((1, 8), dtype=torch.float64)
    observation[:, 6:8] = torch.tensor([[0.0, 0.25]], dtype=torch.float64)
    predicted_yaw = torch.tensor(
        [[error / GO2_PARKOUR_YAW_SCALE, -0.5 / GO2_PARKOUR_YAW_SCALE]],
        dtype=torch.float64,
    )

    mixed, use_prediction = apply_parkour_mts(observation, predicted_yaw)

    assert use_prediction.item() is expected_prediction
    expected_yaw = (
        predicted_yaw * GO2_PARKOUR_YAW_SCALE
        if expected_prediction
        else observation[:, 6:8]
    )
    torch.testing.assert_close(mixed[:, 6:8], expected_yaw)


def test_parkour_mts_compares_yaw_across_pi_boundary():
    observation = torch.zeros((1, 8), dtype=torch.float64)
    observation[:, 6] = torch.pi - 0.1
    predicted_yaw = torch.tensor(
        [[(-torch.pi + 0.1) / GO2_PARKOUR_YAW_SCALE, 0.0]],
        dtype=torch.float64,
    )

    mixed, use_prediction = apply_parkour_mts(observation, predicted_yaw)

    assert use_prediction.item()
    torch.testing.assert_close(
        mixed[:, 6:8], predicted_yaw * GO2_PARKOUR_YAW_SCALE
    )
    assert GO2_PARKOUR_MTS_THRESHOLD_RAD == 0.6
