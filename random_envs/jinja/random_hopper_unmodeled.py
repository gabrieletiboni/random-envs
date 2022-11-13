"""
    Under-modeled Hopper environment (we wrongly model the first mass and hope DR can make up for it):

    - 3 hopper masses are randomized
    - the first mass is held fixed to an offset value of -1.0 w.r.t. the original value
"""

import numpy as np
import gym
from gym import utils
from random_envs.jinja.jinja_mujoco_env import MujocoEnv
import pdb
from copy import deepcopy
import csv
from scipy.stats import truncnorm


class RandomHopperUnmodeledEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.original_lengths = np.array([.4, .45, 0.5, .39])
        self.model_args = {"size": list(self.original_lengths)}
        
        MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        
        # Torso mass has an offset value of -20%
        self.original_fixed_mass = self.sim.model.body_mass[1]
        self.sim.model.body_mass[1] = 0.8*self.original_fixed_mass

        # Dynamics parameters to randomize: 3 masses
        self.original_masses = self.sim.model.body_mass[2:]
        self.task_dim = self.original_masses.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dynamics_indexes = {0: 'thighmass', 1: 'legmass', 2: 'footmass'}

        self.preferred_lr = 0.0005 # --algo Sac -t 5M
        self.reward_threshold = 1750


    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               'thighmass': (0.5, 10.0),
               'legmass': (0.5, 10.0),
               'footmass': (0.5, 10.0),
        }
        return search_bounds_mean[self.dynamics_indexes[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'thighmass': 0.001,
                    'legmass': 0.001,
                    'footmass': 0.001
        }

        return lowest_value[self.dynamics_indexes[index]]


    def get_task(self):
        masses = np.array( self.sim.model.body_mass[2:] )
        return masses

    def set_task(self, *task):
        self.sim.model.body_mass[2:] = task


    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        if self.endless:
            done = False

        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
            # np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def get_full_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        return mjstate

    def get_initial_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        return mjstate

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()


gym.envs.register(
        id="RandomHopperUnmodeled-v0",
        entry_point="%s:RandomHopperUnmodeledEnv" % __name__,
        max_episode_steps=500,
)