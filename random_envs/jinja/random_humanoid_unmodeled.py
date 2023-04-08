"""Implementation of the Humanoid environment supporting
domain randomization optimization.

Randomizations:
    - 13 mass links
    - 17 joint damping

First 45 dims in state space are qpos and qvel

Unmodeled:
    - 4 masses
    - 3 joint damping

For all details: https://www.gymlibrary.ml/environments/mujoco/humanoid/
"""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from random_envs.jinja.jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class RandomHumanoidUnmodeled(MujocoEnv, utils.EzPickle):
    def __init__(self):
        # self.original_lengths = np.array([.4, .45, 0.5, .39])
        # self.model_args = {"size": list(self.original_lengths)}

        # self.noisy = noisy
        # # Rewards:
        # #   noiseless: 
        # #   1e-5: 2554 +- 400
        # #   1e-4: 2600 +- 250
        # #   1e-3: 2652 +- 223
        # #   1e-2: 
        # self.noise_level = 1e-3

        MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

        self.original_fixed_mass = self.sim.model.body_mass[1:5]
        self.sim.model.body_mass[1] = 0.8*self.original_fixed_mass[0]
        self.sim.model.body_mass[2] = 0.8*self.original_fixed_mass[1]
        self.sim.model.body_mass[3] = 0.8*self.original_fixed_mass[2]
        self.sim.model.body_mass[4] = 0.8*self.original_fixed_mass[3]

        self.original_fixed_damping = self.sim.model.dof_damping[6:9]
        self.sim.model.dof_damping[6] = 0.8*self.original_fixed_damping[0]
        self.sim.model.dof_damping[7] = 0.8*self.original_fixed_damping[1]
        self.sim.model.dof_damping[8] = 0.8*self.original_fixed_damping[2]

        self.original_masses = np.copy(self.sim.model.body_mass[5:])
        self.original_damping = np.copy(self.sim.model.dof_damping[9:])

        self.task_dim = self.original_masses.shape[0] + self.original_damping.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {#0: 'mass0', 1: 'mass1', 2: 'mass2', 3: 'mass3',
                                 0: 'mass4', 1: 'mass5', 2: 'mass6', 3: 'mass7',
                                 4: 'mass8', 5: 'mass9', 6: 'mass10', 7: 'mass11', 8: 'mass12',
                                 #10: 'damp1', 12: 'damp2', 13: 'damp3',
                                 9: 'damp4', 10: 'damp5',
                                 11: 'damp6', 12: 'damp7', 13: 'damp8', 14: 'damp9', 15: 'damp10',
                                 16: 'damp11', 17: 'damp12',  18: 'damp13', 19: 'damp14', 20: 'damp15',
                                 21: 'damp16', 22: 'damp17'}

        self.preferred_lr = 0.0001 # --algo Sac -t 5M
        self.reward_threshold = 2200



    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               # 'mass0': (0.5, 10.0),
               # 'mass1': (0.5, 10.0),
               # 'mass2': (0.5, 10.0),
               # 'mass3': (0.5, 10.0),
               'mass4': (0.5, 10.0),
               'mass5': (0.5, 10.0),
               'mass6': (0.5, 10.0),
               'mass7': (0.5, 10.0),
               'mass8': (0.5, 10.0),
               'mass9': (0.5, 10.0),
               'mass10': (0.5, 10.0),
               'mass11': (0.5, 10.0),
               'mass12': (0.5, 10.0),

               # 'damp1': (1, 10.0),
               # 'damp2': (1, 10.0),
               # 'damp3': (1, 10.0),
               'damp4': (1, 10.0),
               'damp5': (1, 10.0),
               'damp6': (1, 10.0),
               'damp8': (1, 10.0),
               'damp9': (1, 10.0),
               'damp10': (1, 10.0),

               'damp7': (.2, 5.0),
               'damp11': (.2, 5.0),
               'damp12': (.2, 5.0),
               'damp13': (.2, 5.0),
               'damp14': (.2, 5.0),
               'damp15': (.2, 5.0),
               'damp16': (.2, 5.0),
               'damp17': (.2, 5.0),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    # 'mass0': 0.2,
                    # 'mass1': 0.2,
                    # 'mass2': 0.2,
                    # 'mass3': 0.2,
                    'mass4': 0.2,
                    'mass5': 0.2,
                    'mass6': 0.2,
                    'mass7': 0.2,
                    'mass8': 0.2,
                    'mass9': 0.2,
                    'mass10': 0.2,
                    'mass11': 0.2,
                    'mass12': 0.2,

                    # 'damp1': 0.8,
                    # 'damp2': 0.8,
                    # 'damp3': 0.8,
                    'damp4': 0.8,
                    'damp5': 0.8,
                    'damp6': 0.8,
                    'damp8': 0.8,
                    'damp9': 0.8,
                    'damp10': 0.8,

                    'damp7': .15,
                    'damp11': .15,
                    'damp12': .15,
                    'damp13': .15,
                    'damp14': .15,
                    'damp15': .15,
                    'damp16': .15,
                    'damp17': .15,
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = np.array( self.sim.model.body_mass[5:] )
        damping = np.array( self.sim.model.dof_damping[9:]  )
        return np.append(masses, damping)

    def set_task(self, *task):
        self.sim.model.body_mass[5:] = task[:9]
        self.sim.model.dof_damping[9:] = task[9:]


    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        if self.endless:
            done = False

        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def _get_obs(self):
        data = self.sim.data

        return np.concatenate(
                [
                data.qpos.flat[2:], # 22
                data.qvel.flat, # 23
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )


    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )

        if self.dr_training:
            self.set_random_task() # Sample new dynamics
            
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


    def get_full_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1] = 0.
        mjstate.qpos[2:] = state[:22] # 24 qpos dim (rootx and rooty ignored)
        mjstate.qvel[:] = state[22:(22+23)] # 23 qvel dim

        return mjstate

    def get_initial_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1] = 0.
        mjstate.qpos[2:] = state[:22] # 24 qpos dim (rootx and rooty ignored)
        mjstate.qvel[:] = state[22:(22+23)] # 23 qvel dim

        return mjstate

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()


gym.envs.register(
        id="RandomHumanoidUnmodeled-v0",
        entry_point="%s:RandomHumanoidUnmodeled" % __name__,
        max_episode_steps=500
)