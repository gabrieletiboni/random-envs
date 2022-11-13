"""Implementation of the HalfCheetah environment supporting
domain randomization optimization.

Randomizations:
    - 7 mass links
    - 1 friction coefficient (sliding)

Unmodeled:
    - 3 masses

For all details: https://www.gymlibrary.ml/environments/mujoco/half_cheetah/
"""
import numpy as np
import gym
from gym import utils
from random_envs.jinja.jinja_mujoco_env import MujocoEnv
from copy import deepcopy
import pdb

class RandomHalfCheetahUnmodeled(MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.original_lengths = np.array([1., .15, .145, .15, .094, .133, .106, .07])
        self.model_args = {"size": list(self.original_lengths)}

        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        self.original_fixed_mass = self.sim.model.body_mass[1:4]
        self.sim.model.body_mass[1] = 0.8*self.original_fixed_mass[0]
        self.sim.model.body_mass[2] = 0.8*self.original_fixed_mass[1]
        self.sim.model.body_mass[3] = 0.8*self.original_fixed_mass[2]

        self.original_masses = np.copy(self.sim.model.body_mass[4:])
        self.original_friction = np.array([0.4])
        self.task_dim = self.original_masses.shape[0] + self.original_friction.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dynamics_indexes = {0: 'bfoot', 1: 'fthigh', 2: 'fshin', 3: 'ffoot', 4: 'friction' }

        self.preferred_lr = 0.0005 # --algo Sac -t 5M
        self.reward_threshold = 4500

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               # 'torso': (0.5, 10.0),
               # 'bthigh': (0.5, 10.0),
               # 'bshin': (0.5, 10.0),
               'bfoot': (0.5, 10.0),
               'fthigh': (0.5, 10.0),
               'fshin': (0.5, 10.0),
               'ffoot': (0.5, 10.0),
               'friction': (0.1, 2.0),
        }
        return search_bounds_mean[self.dynamics_indexes[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    # 'torso': 0.1,
                    # 'bthigh': 0.1,
                    # 'bshin': 0.1,
                    'bfoot': 0.1,
                    'fthigh': 0.1,
                    'fshin': 0.1,
                    'ffoot': 0.1,
                    'friction': 0.02,
        }

        return lowest_value[self.dynamics_indexes[index]]

    def get_task(self):
        masses = np.array( self.sim.model.body_mass[4:] )
        friction = np.array( self.sim.model.pair_friction[0,0] )
        task = np.append(masses, friction)
        return task

    def set_task(self, *task):
        # self.current_lengths = np.array(task[-len(self.original_lengths):])
        # self.model_args = {"size": list(self.current_lengths)}
        # self.build_model()
        # self.sim.model.body_mass[1:] = task[:-len(self.original_lengths)]

        self.sim.model.body_mass[4:] = task[:-1]
        self.sim.model.pair_friction[0:2,0:2] = task[-1]


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_full_mjstate(self, state, template):
        mjstate = deepcopy(template)
       
        mjstate.qpos[0] = 0.   # We *don't know* the x coordinate. But it doesn't affect the behaviour.
        mjstate.qpos[1:] = state[:8]   # Write other positions
        mjstate.qvel[:] = state[8:]   # Write velocities

        return mjstate

    def get_initial_mjstate(self, state, template):
        mjstate = deepcopy(template)
       
        mjstate.qpos[0] = 0.   # We *don't know* the x coordinate. But it doesn't affect the behaviour.
        mjstate.qpos[1:] = state[:8]   # Write other positions
        mjstate.qvel[:] = state[8:]   # Write velocities

        return mjstate

    def set_sim_state(self, mjstate):
        mjstate = self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()


gym.envs.register(
        id="RandomHalfCheetahUnmodeled-v0",
        entry_point="%s:RandomHalfCheetahUnmodeled" % __name__,
        max_episode_steps=500
)