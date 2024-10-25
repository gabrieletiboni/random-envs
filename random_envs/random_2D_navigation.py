import pdb
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import pygame

from random_envs.random_env import RandomEnv
from .random_2D_nav_utils.geometry import Random2DNavigationBox



class Random2DNavigation(RandomEnv):
    def __init__(
        self,
        vertical_wind=False,
        isd_randomness=None,
        isd_random_vel=False,
        init_pos_distr_fraction_h=0.0,
        init_pos_distr_fraction_v=0.0,
        init_pos_distr_fraction_vel_v=0.0,
        init_pos_distr_fraction_vel_h=0.0,
    ):
        """
        Setting isd_randomness overwrites the dimension-specific arguments
        """
        RandomEnv.__init__(self)

        # Define the observation space (width, hight, h_vel, v_vel)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Define the action space (applied horizontal and vertical force)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Initialize the box position and other variables
        self.initial_v_offset = 0.1
        self.box_pos = np.array([0.0, self.initial_v_offset], dtype=np.float32)
        self.box_vel = np.array([0.0, 0.0], dtype=np.float32)

        self.timestep = 0.05
        self.box_mass = 1.0
        self.max_force = 2.0
        self.goal = np.array([0.0, 1.1], dtype=np.float32)  # Position of goal
        self.wind = np.array(
            [0.0, 0.0], dtype=np.float32
        )  # Default force from wind (no force)

        self.dyn_ind_to_name = (
            {0: "horizontal_wind_force", 1: "vertical_wind_force"}
            if vertical_wind
            else {0: "horizontal_wind_force"}
        )

        self.seed()
        self.viewer = None

        self.task_dim = len(self.dyn_ind_to_name.keys())
        self.original_task = np.array(self.wind[: self.task_dim])
        self.nominal_values = np.copy(self.original_task)
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.preferred_lr = None
        self.reward_threshold = 0  # temp

        # let's make a box out of rectangles
        self.bounding_box = Random2DNavigationBox()

        # this should be controlled
        if isd_randomness is not None:
            init_pos_distr_fraction_h = isd_randomness
            init_pos_distr_fraction_v = isd_randomness
            if isd_random_vel:
                init_pos_distr_fraction_vel_v = isd_randomness
                init_pos_distr_fraction_vel_h = isd_randomness
        self.init_box_pos_distr = self.bounding_box.get_scaled_area_before_wall(
            self.bounding_box.get_area_before_wall(),
            init_pos_distr_fraction_h,
            init_pos_distr_fraction_v,
            self.initial_v_offset
        )

        # quick one liner, of course the same thing done for position should be done for velocity, ie
        # separate h and v calculations
        self.init_box_vel_distr = np.array([-0.5,0.5,-0.5,0.5])*init_pos_distr_fraction_vel_h

        self.game_display = None
        self.clock = None

        self.verbose = 0

    def reset(self):
        # Sample new dynamics
        if self.dr_training:
            self.set_random_task()

        # Reset box position and velocity
        self.box_pos = np.array(
            [
                np.random.uniform(
                    low=self.init_box_pos_distr[0], high=self.init_box_pos_distr[1]
                ),
                np.random.uniform(
                    low=self.init_box_pos_distr[2], high=self.init_box_pos_distr[3]
                ),
            ],
            dtype=np.float32,
        )
        self.box_vel = np.array(
            [
                np.random.uniform(
                    low=self.init_box_vel_distr[0], high=self.init_box_vel_distr[1]
                ),
                np.random.uniform(
                    low=self.init_box_vel_distr[2], high=self.init_box_vel_distr[3]
                ),
            ],
            dtype=np.float32,
        )

        # Reset distance from goal
        self.distance_from_goal = self.get_distance(self.box_pos, self.goal)

        return self._get_state()

    def step(self, action):
        # Update the box position based on the applied force and gravity
        input_force = action * self.max_force
        total_force = input_force + self.wind

        acceleration = total_force / self.box_mass

        self.box_vel = self.box_vel + acceleration * self.timestep

        has_hit_wall = self.bounding_box.does_trajectory_hit(
            self.box_pos,
            self.box_vel,
            np.array([0.0, 0.0], dtype=np.float32),
            self.timestep,
        )

        self.box_pos = self.box_pos + self.box_vel * self.timestep

        reward = self._get_reward(self.box_pos) - (.0 / 100 if has_hit_wall else 0.0)
        done = has_hit_wall
        info = {"distance_from_goal": self.get_distance(self.box_pos, self.goal)}

        return self._get_state(), reward, done, info

    def _get_state(self):
        return np.concatenate((self.box_pos, self.box_vel))

    def _get_reward(self, x):
        d = self.get_squared_distance(x, self.goal)
        return (- d - np.log(d+1e-3) + 0.8) / 100

    def get_distance(self, position, goal):
        return np.sqrt(np.sum((position - goal) ** 2))

    def get_squared_distance(self, position, goal):
        return np.sum((position - goal) ** 2)

    def render(self, mode="human"):
        """Render the scene"""
        L = 800
        P = 100  # padding
        W = (255, 255, 255)
        G = (0, 255, 0)
        B = (0, 0, 0)
        BL = (0, 0, 255)
        SCALE = (L - P) / 1.2

        def t(x, y):
            return L / 2 + x * SCALE, (L - P) - y * SCALE + P / 2

        if self.game_display is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.game_display = pygame.display.set_mode((L, L))
            self.game_font = pygame.font.SysFont('Arial', 10)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.game_display.fill(W)
        for r in self.bounding_box.rectangles:
            left, top = t(r.values[0], r.values[3])
            left, top = int(left), int(top)
            width = int((r.values[1] - r.values[0]) * SCALE)
            height = int((r.values[3] - r.values[2]) * SCALE)
            pygame.draw.rect(self.game_display, B, (left, top, width, height))
        left, top = t(*self.box_pos)
        left, top = int(left), int(top)
        pygame.draw.circle(self.game_display, BL, (left, top), 20)
        left, top = t(*self.goal)
        left, top = int(left), int(top)
        pygame.draw.circle(self.game_display, G, (left, top), 20)
        text_surface = self.game_font.render("wind: " + str(self.wind), False, (0, 0, 0))
        self.game_display.blit(text_surface, (0,0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(24)

    def close(self):
        if self.game_display is not None:
            pygame.display.quit()
            pygame.quit()

    def get_task(self):
        i = 2 if 1 in self.dyn_ind_to_name.keys() else 1
        return np.array(self.wind[:i])

    def set_task(self, *task):
        for i in self.dyn_ind_to_name.keys():
            self.wind[i] = task[i]

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
            "horizontal_wind_force": (-2., 2.),
            "vertical_wind_force": (-2., 2.),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
            "horizontal_wind_force": -2.,
            "vertical_wind_force": -2.,
        }
        return lowest_value[self.dyn_ind_to_name[index]]

    def get_task_upper_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        upper_value = {
            "horizontal_wind_force": 2.,
            "vertical_wind_force": 2.,
        }
        return upper_value[self.dyn_ind_to_name[index]]

    def set_verbosity(self, verbose):
        self.verbose = verbose

gym.envs.register(
    id="Random2DNavigation-v0",
    entry_point="%s:Random2DNavigation" % __name__,
    max_episode_steps=100,
    kwargs={},
)

gym.envs.register(
    id="Random2DNavigation_r25-v0",
    entry_point="%s:Random2DNavigation" % __name__,
    max_episode_steps=100,
    kwargs={
        "init_pos_distr_fraction_h": 0.25,
        "init_pos_distr_fraction_v": 0.25,
    }
)
gym.envs.register(
    id="Random2DNavigation_r50-v0",
    entry_point="%s:Random2DNavigation" % __name__,
    max_episode_steps=100,
    kwargs={
        "init_pos_distr_fraction_h": 0.50,
        "init_pos_distr_fraction_v": 0.50,
    }
)
gym.envs.register(
    id="Random2DNavigation_r75-v0",
    entry_point="%s:Random2DNavigation" % __name__,
    max_episode_steps=100,
    kwargs={
        "init_pos_distr_fraction_h": 0.75,
        "init_pos_distr_fraction_v": 0.75,
    }
)
gym.envs.register(
    id="Random2DNavigation_r100-v0",
    entry_point="%s:Random2DNavigation" % __name__,
    max_episode_steps=100,
    kwargs={
        "init_pos_distr_fraction_h": 1.,
        "init_pos_distr_fraction_v": 1.,
    }
)
gym.envs.register(
    id="Random2DNavigationWithRandomVel-v0",
    entry_point="%s:Random2DNavigation" % __name__,
    max_episode_steps=100,
    kwargs={"isd_random_vel": True},
)