"""Implementation of the Walker2d environment supporting
domain randomization optimization.

Randomizations:
    - 7 masses
    - 4 link legnths (torso, then the 3 links for each leg symmetrically)
    - 2 friction coefficient (sliding), separately for each foot

Unmodeled:
    - 3 masses
    - 1 link length (torso)

    All details: https://www.gymlibrary.ml/environments/mujoco/walker2d/
"""

import numpy as np
import gym
from gym import utils
from random_envs.jinja.jinja_mujoco_env import MujocoEnv
from copy import deepcopy
import pdb

class RandomWalker2dUnmodeled(MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.original_lengths = np.array([.4, .45, 0.6, .2])
        self.original_fixed_length = self.original_lengths[:1]
        self.original_lengths[0] = 0.8*self.original_fixed_length[0]
        self.model_args = {"size": list(self.original_lengths)}

        MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

        self.original_fixed_mass = self.sim.model.body_mass[1:4]
        self.sim.model.body_mass[1] = 0.8*self.original_fixed_mass[0]
        self.sim.model.body_mass[2] = 0.8*self.original_fixed_mass[1]
        self.sim.model.body_mass[3] = 0.8*self.original_fixed_mass[2]

        self.original_masses = self.sim.model.body_mass[4:]
        self.current_lengths = np.array(self.original_lengths)
        self.original_friction = np.array([0.9, 1.9])
        self.task_dim = self.original_masses.shape[0] + (self.current_lengths.shape[0]-1) + self.original_friction.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'foot', 1: 'thigh_left', 2: 'leg_left', 3: 'foot_left', 4: 'thighsize', 5: 'legsize', 6: 'footsize', 7: 'friction_right', 8: 'friction_left'}

        self.preferred_lr = 0.0005
        self.reward_threshold = 2200


    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               # 'torso': (0.5, 10.0),
               # 'thigh': (0.5, 10.0),
               # 'leg': (0.5, 10.0),
               'foot': (0.5, 10.0),
               'thigh_left': (0.5, 10.0),
               'leg_left': (0.5, 10.0),
               'foot_left': (0.5, 10.0),

               # 'torsosize': (0.1, 1.0),
               'thighsize': (0.3, 1.0),
               'legsize': (0.3, 1.0),
               'footsize': (0.15, 0.8),

               'friction_right': (0.1, 3.0),
               'friction_left': (0.1, 3.0)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    # 'torso': 0.1,
                    # 'thigh': 0.1,
                    # 'leg': 0.1,
                    'foot': 0.1,
                    'thigh_left': 0.1,
                    'leg_left': 0.1,
                    'foot_left': 0.1,

                    # 'torsosize': 0.1,
                    'thighsize': 0.25,
                    'legsize': 0.25,
                    'footsize': 0.12,

                    'friction_right': 0.05,
                    'friction_left': 0.05
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = self.sim.model.body_mass[4:]
        friction = np.array( self.sim.model.pair_friction[0:2,0] )
        return np.concatenate((masses, self.current_lengths[1:], friction))

    def set_task(self, *task):
        # self.current_lengths = np.array(task[-len(self.original_lengths):])
        self.current_lengths[1:] = np.array(task[len(self.original_masses):len(self.original_masses)+len(self.original_lengths)-1])
        self.model_args = {"size": list(self.current_lengths)}
        self.build_model()
        self.sim.model.body_mass[4:] = task[:len(self.original_masses)]
        self.sim.model.pair_friction[0,0:2] = task[-2]
        self.sim.model.pair_friction[1,0:2] = task[-1]


    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        if self.endless:
            done = False
        
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        obs = np.concatenate([qpos[1:], qvel[:]]).ravel()

        return obs

    def reset_model(self):
        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )        

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def get_full_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:8]
        mjstate.qvel[:] = state[8:]

        return mjstate

    def get_initial_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:8]
        mjstate.qvel[:] = state[8:]

        return mjstate

    def set_sim_state(self, state):
        return self.sim.set_state(state)

    def get_sim_state(self):
        return self.sim.get_state()


gym.envs.register(
        id="RandomWalker2dUnmodeled-v0",
        entry_point="%s:RandomWalker2dUnmodeled" % __name__,
        max_episode_steps=500
)


