"""Single-arm Franka FR3 output transform for TA-VLA.

Inputs reuse TavlaInputs from tavla_policy -- state padding is dim-agnostic and
the canonical camera keys are produced by the data-config repack. Only the
output truncation differs: FR3 actions are 8-dim (7 joints + gripper) instead
of bimanual 14-dim.
"""
import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class Fr3Outputs(transforms.DataTransformFn):
    """Keep the first 8 action dims: joint_1..7 + gripper."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :8])
        return {"actions": actions}
