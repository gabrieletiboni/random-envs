import pdb
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import pygame

from random_envs.random_env import RandomEnv


class Random2DNavigationControlledpos(RandomEnv):
    def __init__(
        self,
        vertical_wind=False,
        isd_randomness=None,
        init_pos_distr_fraction_h=0.0,
        init_pos_distr_fraction_v=0.0,
    ):
        """
        Setting isd_randomness overwrites the dimension-specific arguments
        """
        RandomEnv.__init__(self)

        # Define the observation space (width, hight, h_vel, v_vel)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Define the action space (applied horizontal and vertical force)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Initialize the box position and other variables
        self.initial_v_offset = 0.1
        self.box_pos = np.array([0.0, self.initial_v_offset], dtype=np.float32)

        self.timestep = 0.05
        self.box_mass = 1.0
        self.delta = 0.02
        self.goal = np.array([0.0, 1.1], dtype=np.float32)  # Position of goal
        self.wind = np.array(
            [0.0, 0.0], dtype=np.float32
        )  # Default force from wind (no force)

        self.dyn_ind_to_name = (
            {0: "horizontal_wind_displacement", 1: "vertical_wind_displacement"}
            if vertical_wind
            else {0: "horizontal_wind_displacement"}
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

        # self.wandb_extra_metrics = {'distance_from_goal': 'distance_from_goal'}
        # self.success_metric = 'distance_from_goal'
        # self.distance_from_goal = 1.
        self.hit_wall_epsilon = 0.01
        self.wall_height = 0.9

        # this should be controlled
        if isd_randomness is not None:
            init_pos_distr_fraction_h = isd_randomness
            init_pos_distr_fraction_v = isd_randomness
        h = (0.5 - self.hit_wall_epsilon) * init_pos_distr_fraction_h
        v_t = (
            self.wall_height - self.hit_wall_epsilon - self.initial_v_offset
        ) * init_pos_distr_fraction_v + self.initial_v_offset
        self.init_box_pos_distr = np.array([-h, h, self.initial_v_offset, v_t])

        # let's make a box out of rectangles
        self.bounding_box = Box()
        rectangle = Rectangle.from_line_with_epsilon(
            -0.5, 0.5, 0.0, self.hit_wall_epsilon, variable="x"
        )
        self.bounding_box.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            -0.5, 0.5, 1.2, self.hit_wall_epsilon, variable="x"
        )
        self.bounding_box.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            0.0, 1.2, -0.5, self.hit_wall_epsilon, variable="y"
        )
        self.bounding_box.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            0.0, 1.2, 0.5, self.hit_wall_epsilon, variable="y"
        )
        self.bounding_box.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            -0.5, -0.2, self.wall_height, self.hit_wall_epsilon, variable="x"
        )
        self.bounding_box.add_rectangle(rectangle)
        rectangle = Rectangle.from_line_with_epsilon(
            0.2, 0.5, self.wall_height, self.hit_wall_epsilon, variable="x"
        )
        self.bounding_box.add_rectangle(rectangle)

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

        # Reset distance from goal
        self.distance_from_goal = self.get_distance(self.box_pos, self.goal)

        return self._get_state()

    def step(self, action):
        # Update the box position based on the applied force and gravity
        input_delta = action * self.delta / np.sqrt(np.sum(np.square(action)))
        total_delta = input_delta + self.wind

        has_hit_wall = self.bounding_box.does_line_hit(
            self.box_pos,
            self.box_pos + total_delta,
        )

        self.box_pos = self.box_pos + total_delta

        reward = self._get_reward(self.box_pos) - (0.0 / 100 if has_hit_wall else 0.0)
        done = has_hit_wall
        info = {"distance_from_goal": self.get_distance(self.box_pos, self.goal)}

        return self._get_state(), reward, done, info

    def _get_state(self):
        return np.array(self.box_pos)

    def _get_reward(self, x):
        d = self.get_squared_distance(x, self.goal)
        return (-d - np.log(d + 1e-3) + 0.8) / 100

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
            self.game_font = pygame.font.SysFont('Arial', 15)
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
        text_surface = self.game_font.render("wind: " + str(self.wind), True, (0, 0, 0))
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
        h_w = (-0.02,0.02) if not 1 in self.dyn_ind_to_name.keys() else (-0.014,0.014)
        search_bounds_mean = {
            "horizontal_wind_displacement": h_w,
            "vertical_wind_displacement": (-0.014, 0.014),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        h_w = -0.02 if not 1 in self.dyn_ind_to_name.keys() else -0.014
        lowest_value = {
            "horizontal_wind_displacement": h_w,
            "vertical_wind_displacement": -0.014,
        }
        return lowest_value[self.dyn_ind_to_name[index]]

    def get_task_upper_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        h_w = 0.02 if not 1 in self.dyn_ind_to_name.keys() else 0.014
        upper_value = {
            "horizontal_wind_displacement": h_w,
            "vertical_wind_displacement": 0.014,
        }
        return upper_value[self.dyn_ind_to_name[index]]

    def set_verbosity(self, verbose):
        self.verbose = verbose


class Figure:
    def does_trajectory_hit(self, pos, vel, acc, dt):
        raise NotImplementedError
    
    def does_line_hit(self, pos1, pos2):
        raise NotImplementedError


class HalfPlane(Figure):
    def __init__(self, value, relation="gt", variable="x"):
        self.value = value
        self.relation = 1.0 if relation == "gt" else -1.0
        self.variable = 0 if variable == "x" else 1

    def does_trajectory_hit(self, pos, vel, acc, dt):
        c = (pos[self.variable] - self.value) * self.relation
        b = vel[self.variable] * self.relation
        a = 0.5 * acc[self.variable] * self.relation

        if c >= 0.0:
            return True

        eps = 1e-7
        if abs(b) < eps and abs(a) < eps:
            return False
        if abs(a) < eps:
            zero = -c / b
            return 0 < zero < dt

        delta = np.square(b) - 4 * a * c

        if delta < 0.0:
            return False

        zero_one = (-b + np.sqrt(delta)) / (2 * a)
        zero_two = (-b - np.sqrt(delta)) / (2 * a)

        return 0 < zero_one < dt or 0 < zero_two < dt

    def does_line_hit(self, pos1, pos2):
        if (pos1[self.variable] - self.value) * self.relation <= 0 and np.sign(
            pos1[self.variable] - self.value
        ) == np.sign(pos2[self.variable] - self.value):
            return False
        return True


class Rectangle(Figure):
    def __init__(self, x0, x1, y0, y1):
        """A (axis-aligned) rectangle is stored as a list of half planes and it is
        their intersection"""
        self.planes = [
            HalfPlane(x0, "gt", "x"),
            HalfPlane(x1, "lt", "x"),
            HalfPlane(y0, "gt", "y"),
            HalfPlane(y1, "lt", "y"),
        ]

        self.values = np.array([x0, x1, y0, y1], dtype=np.float32)

    def does_trajectory_hit(self, pos, vel, acc, dt):
        # a rectangle is the intersection of 4 halplanes:
        # it is hit if all the halfplanes are hit (logial and)
        a = True
        for x in self.planes:
            a = a and x.does_trajectory_hit(pos, vel, acc, dt)
        return a
    
    def does_line_hit(self, pos1, pos2):
        a = True
        for x in self.planes:
            a = a and x.does_line_hit(pos1, pos2)
        return a

    @staticmethod
    def from_line_with_epsilon(start, end, other_variable, epsilon, variable="x"):
        assert start < end, "Start position must be strictly lower than end position"
        x0 = start - epsilon
        x1 = end + epsilon
        y0 = other_variable - epsilon
        y1 = other_variable + epsilon
        if variable != "x":
            x0, x1, y0, y1 = y0, y1, x0, x1
        return Rectangle(x0, x1, y0, y1)


class Box(Figure):
    def __init__(self):
        """A union of rectangles"""
        self.rectangles: list[Rectangle] = []

    def add_rectangle(self, rectangle: Rectangle):
        self.rectangles.append(rectangle)

    def does_trajectory_hit(self, pos, vel, acc, dt):
        # a box is a union of rectangles:
        # it is hit if at least one rectangle is hit (logical or)
        a = False
        for x in self.rectangles:
            a = a or x.does_trajectory_hit(pos, vel, acc, dt)
        return a

    def does_line_hit(self, pos1, pos2):
        a = False
        for x in self.rectangles:
            a = a or x.does_line_hit(pos1, pos2)
        return a


gym.envs.register(
    id="Random2DNavigationControlledpos-v0",
    entry_point="%s:Random2DNavigationControlledpos" % __name__,
    max_episode_steps=100,
    kwargs={},
)