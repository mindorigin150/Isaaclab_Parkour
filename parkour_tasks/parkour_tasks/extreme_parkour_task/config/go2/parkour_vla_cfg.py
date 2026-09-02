"""RGB-enabled evaluation configuration for the Parkour VLA benchmark."""

import torch
from parkour_isaaclab.actor import (
    GO2_PARKOUR_PERCEPTION_DIM,
    GO2_PARKOUR_PROPRIO_DIM,
    GO2_PARKOUR_TERRAIN_LATENT_DIM,
    GO2_PARKOUR_YAW_DIM,
)

import isaaclab.sim as sim_utils
from isaaclab.managers import TerminationTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from parkour_tasks.default_cfg import quat_from_euler_xyz_tuple

from .parkour_teacher_cfg import (
    ParkourTeacherSceneCfg,
    UnitreeGo2TeacherParkourEnvCfg_EVAL,
)


PARKOUR_VLA_PROMPT = "Traverse the parkour course and follow each waypoint."
PARKOUR_VLA_CAMERA_PERIOD_S = 0.1
PARKOUR_VLA_LATENT_DIM = GO2_PARKOUR_TERRAIN_LATENT_DIM
PARKOUR_VLA_YAW_DIM = GO2_PARKOUR_YAW_DIM
PARKOUR_VLA_ACTION_DIM = GO2_PARKOUR_PERCEPTION_DIM
PARKOUR_VLA_CONTROL_REPEAT = 5
PARKOUR_VLA_PROPRIO_DIM = GO2_PARKOUR_PROPRIO_DIM
PARKOUR_VLA_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/vla_camera",
    update_period=PARKOUR_VLA_CAMERA_PERIOD_S,
    height=150,
    width=256,
    data_types=["rgb", "instance_segmentation_fast"],
    colorize_instance_segmentation=False,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=11.041,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=12.240,
        clipping_range=(0.1, 100.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.33, 0.0, 0.08),
        rot=quat_from_euler_xyz_tuple(
            *tuple(torch.deg2rad(torch.tensor([180, 70, -90])))
        ),
        convention="ros",
    ),
)


def parkour_success(env) -> torch.Tensor:
    parkour = env.parkour_manager.get_term("base_parkour")
    return parkour.cur_goal_idx >= parkour.num_goals


@configclass
class ParkourVlaSceneCfg(ParkourTeacherSceneCfg):
    """Teacher scene with a batched robot-mounted RGB camera."""

    vla_camera = PARKOUR_VLA_CAMERA_CFG


@configclass
class UnitreeGo2ParkourVlaEnvCfg(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    """Zero-delay Parkour evaluation scene used by collectors and VLA policies."""

    scene: ParkourVlaSceneCfg = ParkourVlaSceneCfg(num_envs=50, env_spacing=1.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.rerender_on_reset = True
        self.terminations.parkour_success = TerminationTermCfg(
            func=parkour_success,
            time_out=False,
        )
        self.scene.vla_camera.update_period = PARKOUR_VLA_CAMERA_PERIOD_S
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.parkours.base_parkour.debug_vis = False
        self.commands.base_velocity.debug_vis = False
