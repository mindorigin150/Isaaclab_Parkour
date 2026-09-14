"""W&B logging on the original teacher's sampling scale."""

from rsl_rl.utils.wandb_utils import WandbSummaryWriter


class TeacherStepWandbWriter(WandbSummaryWriter):
    """Map cumulative control steps to the existing teacher run's Step axis."""

    # The reference teacher collects 6144 environments * 24 control steps/update.
    teacher_control_steps_per_update = 6144 * 24
    control_steps = 0

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        # Ceiling keeps partial batches at their current teacher update position.
        step = (self.control_steps - 1) // self.teacher_control_steps_per_update
        super().add_scalar(tag, scalar_value, step, walltime, new_style)
