"""openbot_sim — a from-scratch simulation of the OpenBot smartphone robot.

The observation/action interface matches the real OpenBot stack so that
trained TensorFlow Lite models are drop-in compatible with the Android app.
"""

from .camera import Camera
from .dynamics import DiffDrive
from .env import OpenBotEnv
from .observation import make_observation
from .policies import RandomPolicy, TeacherPolicy, TFLitePolicy
from .state import State
from .task import LaneKeepingTask
from .world import World

__all__ = [
    "OpenBotEnv",
    "State",
    "DiffDrive",
    "World",
    "Camera",
    "LaneKeepingTask",
    "make_observation",
    "RandomPolicy",
    "TeacherPolicy",
    "TFLitePolicy",
]
