"""Implementation of the Hopper environment supporting
domain randomization optimization.

For all details: https://www.gymlibrary.ml/environments/mujoco/hopper/
"""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from random_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class RandomHopperEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, noisy=False, isd_randomness=0.0):
        self.original_lengths = np.array([.4, .45, 0.5, .39])
        self.model_args = {"size": list(self.original_lengths)}

        self.noisy = noisy
        # Rewards:
        #   noiseless: 2049 
        #   1e-5: 1979 
        #   1e-4: 1835
        #   1e-3: 1591
        #   1e-2: 695
        self.noise_level = 1e-4

        MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.nominal_values = np.concatenate([self.original_masses])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'torsomass', 1: 'thighmass', 2: 'legmass', 3: 'footmass'}

        self.preferred_lr = 0.0005 # --algo Sac -t 5M
        self.reward_threshold = 1750

        self.isd_randomness = isd_randomness
        

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               'torsomass': (0.1, 10.0),
               'thighmass': (0.1, 10.0),
               'legmass': (0.1, 10.0),
               'footmass': (0.1, 10.0),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torsomass': 0.001,
                    'thighmass': 0.001,
                    'legmass': 0.001,
                    'footmass': 0.001
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_task(self, *task):
        self.sim.model.body_mass[1:] = task


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
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
            # np.clip(self.sim.data.qvel.flat, -10, 10)
        ])
        
        if self.noisy:
            obs += np.sqrt(self.noise_level)*np.random.randn(obs.shape[0])

        return obs

    def reset_model(self):
        low = self.isd_randomness * -0.1
        high = self.isd_randomness * 0.5
        qpos = self.init_qpos + self.np_random.uniform(low=low, high=high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=low, high=high, size=self.model.nv)
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
        id="RandomHopperIsd-v0",
        entry_point="%s:RandomHopperEnv" % __name__,
        max_episode_steps=500
)

gym.envs.register(
        id="RandomHopperNoisyIsd-v0",
        entry_point="%s:RandomHopperEnv" % __name__,
        max_episode_steps=500,
        kwargs={"noisy": True}
)