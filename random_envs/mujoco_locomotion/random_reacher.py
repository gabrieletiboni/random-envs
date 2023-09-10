"""Implementation of the Reacher environment supporting
domain randomization optimization.

For all details: https://www.gymlibrary.ml/environments/mujoco/reacher/
"""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from random_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class RandomReacherEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = None
        default_gravity = np.mean(list(self.get_search_bounds_mean(name='gravityx')))
        self.original_gravity = np.array([default_gravity, default_gravity])
        self.current_gravity = self.original_gravity
        self.model_args = {"gravity": list(self.original_gravity)}

        MujocoEnv.__init__(self, 'reacher.xml', 4)
        utils.EzPickle.__init__(self)

        # Set default values in the middle of the search space
        default_mass = np.mean(list(self.get_search_bounds_mean(name='body0mass')))
        default_damping = np.mean(list(self.get_search_bounds_mean(name='damping0')))
        self.set_task(*[default_mass, default_mass, default_damping, default_damping, default_gravity, default_gravity])

        self.original_masses = np.copy(self.sim.model.body_mass[1:3])
        self.original_damping = np.copy(self.sim.model.dof_damping[:2])

        self.nominal_values = np.concatenate([self.original_masses, self.original_damping, self.original_gravity])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'body0mass', 1: 'body1mass', 2: 'damping0', 3: 'damping1', 4: 'gravityx', 5: 'gravityy'}

        self.preferred_lr = 0.001
        self.reward_threshold = 0  # temp
        

    def get_search_bounds_mean(self, index=-1, name=None):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               'body0mass': (0.005, 1),
               'body1mass': (0.005, 1),
               'damping0': (0.01, 50),
               'damping1': (0.01, 50),
               'frictionloss0': (0.01, 10),
               'frictionloss1': (0.01, 10),
               'gravityx': (-100, 100),
               'gravityy': (-100, 100)
        }
        if name is None:
            return search_bounds_mean[self.dyn_ind_to_name[index]]
        else:
            return search_bounds_mean[name]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'body0mass': 0.001,
                    'body1mass': 0.001,
                    'damping0':  0.001,
                    'damping1':  0.001,
                    'frictionloss0': 0.,
                    'frictionloss1': 0.,
                    'gravityx': -1000,
                    'gravityy': -1000
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = np.array( self.sim.model.body_mass[1:3] )
        damping = np.array( self.sim.model.dof_damping[:2]  )
        return np.concatenate([masses, damping, self.current_gravity])

    def set_task(self, *task):
        self.current_gravity = task[-2:]
        self.model_args = {"gravity": list(self.current_gravity)}
        self.build_model()

        self.sim.model.body_mass[1:3] = task[:2]
        self.sim.model.dof_damping[:2] = task[2:4]


    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_current_goal_3d()
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                # self.sim.data.qpos.flat[2:],
                self.get_current_goal(),
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_current_goal_3d(),
            ]
        )

    def reset_model(self):
        # Before potentially re-building the model
        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        self.goal = self.get_random_goal()
        goal_body_id = self.sim.model.body_name2id("target")
        self.sim.model.body_pos[goal_body_id][:2] = self.goal
            
        return self._get_obs()

    def get_random_goal(self):
        while True:
            goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(goal) < 0.2:
                break
        return goal

    def get_current_goal_3d(self):
        goal_body_id = self.sim.model.body_name2id("target")
        return self.sim.model.body_pos[goal_body_id][:]

    def get_current_goal(self):
        if self.goal is None:
            goal_body_id = self.sim.model.body_name2id("target")
            return self.sim.model.body_pos[goal_body_id][:2]
        else:
            return self.goal

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0



gym.envs.register(
        id="RandomReacher-v0",
        entry_point="%s:RandomReacherEnv" % __name__,
        max_episode_steps=100
)