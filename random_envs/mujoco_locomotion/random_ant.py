"""Implementation of the Ant environment supporting
domain randomization optimization.

For all details: https://www.gymlibrary.ml/environments/mujoco/ant/
"""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from random_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class RandomAntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        # self.original_lengths = np.array([.1, .1, .1])
        # self.original_viscosity = 0.1
        # self.model_args = {"size": list(self.original_lengths), 'viscosity': self.original_viscosity}

        MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.get_masses())
        self.original_friction = np.array([1.])
        self.nominal_values = np.concatenate([self.original_masses, self.original_friction])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'torsomass',
                                1: 'leg1mass', 2: 'ankle1mass',
                                3: 'leg2mass', 4: 'ankle2mass',
                                5: 'leg3mass', 6: 'ankle3mass',
                                7: 'leg4mass', 8: 'ankle4mass',
                                9: 'friction'}

        self.preferred_lr = 0.0005 # --algo Sac -t 5M
        self.reward_threshold = 1750
        

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               'torsomass': (0.05, None),
               'leg1mass': (0.01, None),
               'ankle1mass': (0.02, None),
               'leg2mass': (0.01, None),
               'ankle2mass': (0.02, None),
               'leg3mass': (0.01, None),
               'ankle3mass': (0.02, None),
               'leg4mass': (0.01, None),
               'ankle4mass': (0.02, None),
               'friction': (0.05, None)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]


    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
            'torsomass':  0.01,
            'leg1mass':   0.01,
            'ankle1mass': 0.02,
            'leg2mass':   0.01,
            'ankle2mass': 0.02,
            'leg3mass':   0.01,
            'ankle3mass': 0.02,
            'leg4mass':   0.01,
            'ankle4mass': 0.02,
            'friction': 0.01
        }

        return lowest_value[self.dyn_ind_to_name[index]]

    def get_masses(self):
        """Return torso mass and leg and ankle mass for each leg.
        An additional auxiliary geom is used to link the leg to the
        ant torso, but this is always considered to be equal to the leg mass"""
        masses = [self.sim.model.body_mass[1]]  # torso
        
        # 3 geoms per leg, but two are always set to be equal. Retain only two per leg
        for i in range(4):  # for each leg
            masses += [self.sim.model.body_mass[2 + (i*3 + 1)], self.sim.model.body_mass[2 + (i*3 + 2)]]
        
        return masses

    def set_masses(self, masses):
        self.sim.model.body_mass[1] = masses[0]
        for i in range(4):
            self.sim.model.body_mass[2 + i*3] = masses[1 + i*2]
            self.sim.model.body_mass[2 + i*3+1] = masses[1 + i*2]
            self.sim.model.body_mass[2 + i*3+2] = masses[1 + i*2 + 1]

    def get_task(self):
        masses = self.get_masses()
        friction = np.array( [self.sim.model.pair_friction[0, 0]] )
        return np.concatenate((masses, friction))


    def set_task(self, *task):
        self.set_masses(task[:9])
        self.sim.model.pair_friction[:, :2] = np.repeat(task[9], 8).reshape(4,2)
        return


    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        not_terminated = (
            np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        )
        terminated = not not_terminated
        ob = self._get_obs()

        return (
            ob,
            reward,
            terminated,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        return np.concatenate(
                    [
                        self.sim.data.qpos.flat[2:],
                        self.sim.data.qvel.flat,
                        np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                    ]
                )


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        return self._get_obs()


    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5



gym.envs.register(
    id="RandomAnt-v0",
    entry_point="%s:RandomAntEnv" % __name__,
    max_episode_steps=500
)