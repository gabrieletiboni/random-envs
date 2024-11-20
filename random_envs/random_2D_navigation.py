import pdb
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from random_envs.random_env import RandomEnv
from .random_2D_nav_utils.geometry import Random2DNavigationBox
from .random_2D_nav_utils.rendering import Random2DNavRenderer
from .random_2D_nav_utils.dynamics import Random2DNavigationDynamics
from .random_2D_nav_utils.dynamics import Random2DNavigationWithRandomvelDynamics
from .random_2D_nav_utils.dynamics import Random2DNavigationOnlyposDynamics
from .random_2D_nav_utils.dynamics import Random2DNavigationControlledposDynamics
from .random_2D_nav_utils.dynamics import Random2DNavigationControlledposCircularWindDynamics
from .random_2D_nav_utils.dynamics import Random2DNavigationControlledposHighDRDynamics


class Random2DNavigation(RandomEnv):
    def __init__(
        self,
        isd_randomness=None,
    ):
        RandomEnv.__init__(self)
        self.dynamics = self._build_dynamics()
        self.observation_space = self.dynamics.observation_space
        self.action_space = self.dynamics.action_space
        self.dyn_ind_to_name = self.dynamics.dyn_ind_to_name
        self.seed()
        self.viewer = None
        self.preferred_lr = None
        self.reward_threshold = 0  # TODO
        self.bounding_box = Random2DNavigationBox()
        self.dynamics.set_isd_randomness(isd_randomness)
        self.dynamics.set_max_isd_area(self.bounding_box.get_area_before_wall())
        self.game_renderer = None
        self.verbose = 0

        # DROPO stuff
        self.task_dim = self.dynamics.get_task_dim()
        self.original_task = self.dynamics.get_original_task()
        self.nominal_values = np.copy(self.original_task)
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)
    
    def _build_dynamics(self):
        return Random2DNavigationDynamics()
    
    def get_actor_state_mask(self):
        return self.dynamics.get_actor_state_mask()

    def reset(self):
        # Sample new dynamics
        if self.dr_training:
            self.set_random_task()

        return self.dynamics.reset()

    def step(self, action):
        return self.dynamics.step(action, self.bounding_box)

    def render(self, mode="human"):
        if self.game_renderer is None:
            self.game_renderer = Random2DNavRenderer()
        self.game_renderer.render(self.bounding_box, self.dynamics)

    def close(self):
        if self.game_renderer is not None:
            self.game_renderer.close()

    def get_task(self):
        return self.dynamics.get_task()

    def set_task(self, *task):
        self.dynamics.set_task(*task)

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        return self.dynamics.get_search_bounds_mean(index)

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        return self.dynamics.get_task_upper_bound(index)

    def get_task_upper_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        return self.dynamics.get_task_upper_bound(index)

    def set_verbosity(self, verbose):
        self.verbose = verbose
    

class Random2DNavigationWithRandomvel(Random2DNavigation):
    def _build_dynamics(self):
        return Random2DNavigationWithRandomvelDynamics()

class Random2DNavigationOnlypos(Random2DNavigation):
    def _build_dynamics(self):
        return Random2DNavigationOnlyposDynamics()
    
class Random2DNavigationControlledpos(Random2DNavigation):
    def _build_dynamics(self):
        return Random2DNavigationControlledposDynamics()

class Random2DNavigationControlledposCircularWind(Random2DNavigation):
    def _build_dynamics(self):
        return Random2DNavigationControlledposCircularWindDynamics()

class Random2DNavigationControlledposHighDR(Random2DNavigation):
    def _build_dynamics(self):
        return Random2DNavigationControlledposHighDRDynamics()

gym.envs.register(
    id="Random2DNavigation-v0",
    entry_point="%s:Random2DNavigation" % __name__,
    max_episode_steps=100,
    kwargs={},
)

gym.envs.register(
    id="Random2DNavigationWithRandomVel-v0",
    entry_point="%s:Random2DNavigationWithRandomvel" % __name__,
    max_episode_steps=100,
    kwargs={},
)

gym.envs.register(
    id="Random2DNavigationOnlypos-v0",
    entry_point="%s:Random2DNavigationOnlypos" % __name__,
    max_episode_steps=100,
    kwargs={},
)

gym.envs.register(
    id="Random2DNavigationControlledpos-v0",
    entry_point="%s:Random2DNavigationControlledpos" % __name__,
    max_episode_steps=100,
    kwargs={},
)

gym.envs.register(
    id="Random2DNavigationControlledposCircularWind-v0",
    entry_point="%s:Random2DNavigationControlledposCircularWind" % __name__,
    max_episode_steps=100,
    kwargs={},
)

gym.envs.register(
    id="Random2DNavigationControlledposHighDR-v0",
    entry_point="%s:Random2DNavigationControlledposHighDR" % __name__,
    max_episode_steps=100,
    kwargs={},
)