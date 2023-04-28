from typing import Any, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy
import csv
import pdb

import numpy as np
import gym
from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm

from random_envs.mujoco_panda.core.panda_gym_env import PandaGymEnvironment, randomization_setter
from random_envs.mujoco_panda.core.controllers import Controller, \
                                       JointPositionController, \
                                       JointImpedanceController, \
                                       TorqueController
from random_envs.mujoco_panda.core.interpolation import Repeater, LinearInterpolator, QuadraticInterpolator
from random_envs.mujoco_panda.core.utils import env_field, register_panda_env, distance_penalty


class PandaPushEnv(PandaGymEnvironment):
    """
    :description: The simple environment where the Panda robot is manipulating
        a box.
    """
    def __init__(self,
                 model_file,
                 controller,
                 action_interpolator,
                 action_repeat_kwargs,
                 model_kwargs={},
                 controller_kwargs={},
                 render_camera="side_camera",
                 render_res=(320, 240),
                 task_reward="guide",
                 command_type="new_pos",
                 acceleration_penalty_factor=1e-2,
                 limit_power=2,
                 contact_penalties=False,
                 goal_low=np.array([.7, -.3]),
                 goal_high=np.array([1.2, .3]),
                 init_box_low=np.array([0.51, 0]),
                 init_box_high=np.array([0.51, 0.]),
                 push_prec_alpha=1e-3,
                 init_jpos_jitter=0.2,
                 init_jvel_jitter=0.0,
                 rotation_in_obs="none",
                 control_penalty_coeff=3,
                 randomized_dynamics='mf'):
        self.goal_low = goal_low
        self.goal_high = goal_high
        self.init_box_low = init_box_low
        self.init_box_high = init_box_high
        PandaGymEnvironment.__init__(self,
                                     model_file=model_file,
                                     controller=controller,
                                     action_interpolator=action_interpolator,
                                     action_repeat_kwargs=action_repeat_kwargs,
                                     model_kwargs=model_kwargs,
                                     controller_kwargs=controller_kwargs,
                                     render_camera=render_camera,
                                     render_res=render_res,
                                     command_type=command_type,
                                     acceleration_penalty_factor=acceleration_penalty_factor,
                                     limit_power=limit_power,
                                     init_jpos_jitter=init_jpos_jitter,
                                     init_jvel_jitter=init_jvel_jitter)

        # Override observation space, we now also have the box
        # (extra 3pos+3velp+3ori+3velr) and the goal (3pos)
        if rotation_in_obs == "none":
            rot_dims = 0
        elif rotation_in_obs == "rotz":
            rot_dims = 1
        elif rotation_in_obs == "sincosz":
            rot_dims = 2
        else:
            raise ValueError("Invalid rotation_in_obs")

        max_obs = np.array([np.inf]*(self.robot_obs_dim + 6 + rot_dims))
        self.observation_space = gym.spaces.Box(-max_obs, max_obs)
        self.push_prec_alpha = push_prec_alpha

        self.rotation_in_obs = rotation_in_obs

        self.task_reward = task_reward
        self.control_penalty_coeff = control_penalty_coeff  # penalize pos, vel and acc when they are close to the limits
        self.last_dist_from_target = 0  # delta_distance appended to info dict

        self.contact_penalties_enabled = contact_penalties  # penalize contact-pairs proportional to penetration distance
        self.contact_penalties = [("box", "table", 1e2),
                                ("panda0_finger1", "box", 1e2),
                                ("panda0_finger2", "box", 1e2),
                                ("panda0_finger1", "table", 3e7),
                                ("panda0_finger2", "table", 3e7)]
        self.contact_penalties = [(self.sim.model.geom_name2id(c[0]),
                                   self.sim.model.geom_name2id(c[1]),
                                   c[0],
                                   c[1],
                                   c[2])
                                        for c in self.contact_penalties]

        # Set randomized dynamics
        self.dyn_ind_to_name = {}
        self.dyn_type = randomized_dynamics
        self.set_random_dynamics_type(dyn_type=randomized_dynamics)

        self._current_task = np.array(self.get_default_task())

        self.original_task = np.copy(self.get_task())
        self.nominal_values = np.copy(self.original_task)
        self.task_dim = self.get_task().shape[0]
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.reward_threshold = 10000


        # dropo-specific state space
        self.dropo_mode = False

    def get_contacts(self):
        """
        :description: Find contacts included in ``self.contact_penalties`` and return
            their penetration distance times the penalization coefficient. This
            can be used for penalizing contacts between certain geoms in the
            reward function.
        :returns: A dictionary with the keys of the form ``c_name1-name2`` with
            values corresponding to their contact depths scaled by the penalty.
        """
        contact_values = [0] * len(self.contact_penalties)  # initialize contact penalities to zero
        contacts = self.sim.data.contact
        for c in contacts:
            geom1 = c.geom1
            geom2 = c.geom2
            for i, (id1, id2, name1, name2, coeff) in \
                    enumerate(self.contact_penalties):
                if id1 == geom1 and id2 == geom2 or id1 == geom2 and id2 == geom1:  # check whether you want to penalize this pair
                    contact_values[i] += coeff * c.dist**2  # c.dist = penetration distance
                    break
        res = {f"c_{self.contact_penalties[i][2]}-{self.contact_penalties[i][3]}": cv \
                for i, cv in enumerate(contact_values)}
        return res


    def step(self, action):
        state, reward, done, info = super().step(action)

        puck_vel = self.puck_velp
        for dim, vel in zip("xyz", puck_vel):
            info[f"puck_dpos_{dim}"] = vel*self.dt
        info[f"puck_dpos"] = np.sqrt(np.sum(puck_vel**2)) * self.dt

        delta_dist_from_target = self.goal_dist - self.last_dist_from_target
        self.last_dist_from_target = self.goal_dist
        info[f"goal_dist"] = self.goal_dist
        info[f"dgoal_dist"] = delta_dist_from_target

        # Calculate contact penalties
        if self.contact_penalties_enabled:
            contacts = self.get_contacts()
            info = dict(**info, **contacts)
            reward -= np.sum([v for _, v in contacts.items()])

        return state, reward, done, info

    def analyze_contacts(self):
        contacts = self.sim.data.contact
        for c in contacts:
            if c.geom1 == c.geom2 == 0 and c.dist == 0:
                break
            name1 = self.sim.model.geom_id2name(c.geom1)
            name2 = self.sim.model.geom_id2name(c.geom2)
            print(f"{name1} - {name2}: {c.dist}")
        print("-"*80)

    @property
    def puck_orientation(self):
        """
        :return: the orientation of the puck as Euler angles, following
            the package-wide angle convention
        """
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        # qpos is a quaternion, we need Euler angles
        quat = self.sim.data.qpos[puck_joint_id+3:puck_joint_id+7]
        transform = Rotation.from_quat(quat)

        return transform.as_euler(random_envs.mujoco_panda.EULER_ORDER)

    @puck_orientation.setter
    def puck_orientation(self, value):
        """
        :description: Sets the object orientation
        :param value: the new orientation in Euler angles, following the
            package-wide ordering convention
        """
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        # qpos is a quaternion, we need Euler angles
        transform = Rotation.from_euler(random_envs.mujoco_panda.EULER_ORDER, value)
        quat = transform.as_quat()
        self.sim.data.qpos[puck_joint_id+3:puck_joint_id+7] = quat

    @property
    def puck_orientation_quat(self):
        """
        :return: the orientation of the puck as Euler angles, following
            the package-wide angle convention
        """
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        # qpos is a quaternion, we need Euler angles
        quat = self.sim.data.qpos[puck_joint_id+3:puck_joint_id+7]
        # transform = Rotation.from_quat(quat)
        # return transform.as_euler(random_envs.mujoco_panda.EULER_ORDER)
        return quat

    @puck_orientation_quat.setter
    def puck_orientation_quat(self, value):
        """
        :description: Sets the object orientation
        :param value: the new orientation in Euler angles, following the
            package-wide ordering convention
        """
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        # qpos is a quaternion, we need Euler angles
        # transform = Rotation.from_euler(random_envs.mujoco_panda.EULER_ORDER, value)
        # quat = transform.as_quat()
        self.sim.data.qpos[puck_joint_id+3:puck_joint_id+7] = value

    @property
    def puck_velr(self):
        """
        :return: the rotational velocity of the puck. The value is clamped to
            (-10, 10) to prevent training instabilities in case the
            simulation gets unstable.
        """
        value = self.sim.data.get_geom_xvelr("box")
        value = np.clip(value, -10, 10)
        return value

    @puck_velr.setter
    def puck_velr(self, value):
        puck_joint_id = self.sim.model.joint_name2id("puck_joint") + 3
        value_dim = value.shape[0]
        self.sim.data.qvel[puck_joint_id:puck_joint_id+value_dim] = value

    @property
    def puck_velp(self):
        """
        :return: the linear velocity of the puck. The value is clamped to
            (-10, 10) to prevent training instabilities in case
            the simulation gets unstable.
        """
        value = self.sim.data.get_geom_xvelp("box")
        value = np.clip(value, -10, 10)
        return value

    @puck_velp.setter
    def puck_velp(self, value):
        """
        :description: Sets the linear velocity of the object to the given value
        :param value: the new velocity of the puck
        """
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        value_dim = value.shape[0]
        self.sim.data.qvel[puck_joint_id:puck_joint_id+value_dim] = value

    @property
    def puck_pos(self):
        """
        :return: the position of the puck. The value is clamped to (-10, 10)
            to prevent training instabilities in case the simulation gets
            unstable.
        """
        value = self.sim.data.get_geom_xpos("box")
        value = np.clip(value, -10, 10)
        return value

    @puck_pos.setter
    def puck_pos(self, value):
        """
        :description: Moves the puck to a new position. If value is 3D, the XYZ
            coordinates of the puck are changed; if it's 2D, only XY position
            is affected (and Z stays whatever it was).
        :param value: the new position of the puck
        """
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        value_dim = value.shape[0]
        self.sim.data.qpos[puck_joint_id:puck_joint_id+value_dim] = value

    def get_observation(self):
        if not self.dropo_mode:
            if self.rotation_in_obs == "none":
                rot = []
            elif self.rotation_in_obs == "rotz":
                rot = [self.puck_orientation[2]]
            elif self.rotation_in_obs == "sincosz":
                rot = [np.sin(self.puck_orientation[2]), np.cos(self.puck_orientation[2])]

            return np.concatenate([super().get_observation(), self.puck_pos,
                                   rot, self.goal_pos])
        else:
            return np.concatenate([super().get_observation(), self.puck_pos[:3],
                                   self.puck_velp[:3], self.puck_orientation_quat, self.puck_velr])

    def set_dropo_mode(self):
        self.dropo_mode = True

    @property
    def goal_dist(self):
        goal_dist = np.sqrt(np.sum((self.puck_pos - self.goal_pos)**2))
        return goal_dist

    def get_task_reward(self):
        guide_dist = np.sqrt(np.sum((self.puck_pos - self.gripper_pos)**2))

        # Prevent both terms from reaching super low values if things go wrong
        # (it can make the training a bit unstable)
        guide_dist = min(guide_dist, 2)
        goal_dist = min(self.goal_dist, 2)

        guide_term = distance_penalty(guide_dist, alpha=1e-1)
        goal_term = distance_penalty(goal_dist, alpha=self.push_prec_alpha)

        if self.task_reward == "guide":
            return goal_term + 0.1 * guide_term
        elif self.task_reward == "lessguide":
            return goal_term + 0.01 * guide_term
        elif self.task_reward == "moreguide":
            return goal_term + 0.5 * guide_term
        elif self.task_reward == "reach":
            return guide_term
        elif self.task_reward == "target":
            return goal_term
        elif self.task_reward == None:
            return 0
        else:
            raise ValueError(f"Unknown reward type: {self.task_reward}")

    def set_random_goal(self):
        goal = np.random.uniform(self.goal_low, self.goal_high)
        goal_site_id = self.sim.model.site_name2id("goal")
        self.sim.model.site_pos[goal_site_id][:2] = goal

    def reset(self):
        # Sample new dynamics and re_build model if necessary
        if self.dr_training:
            self.set_random_task()

        super().reset()
        self.set_random_goal()
        start_pos = np.random.uniform(self.init_box_low, self.init_box_high)
        self.puck_pos = start_pos
        self.last_dist_from_target = 0
        return self.get_observation()


    def get_puck_friction(self):
        return self.get_pair_friction("box", "table")

    @randomization_setter("friction")
    def set_puck_friction(self, value):
        """
        :description: Sets the friction between the puck and the sliding surface
        :param value: New friction value. Can either be an array of 2 floats
            (to set the linear friction) or an array of 5 float (to set the
            torsional and rotational friction values as well)
        :raises ValueError: if the dim of ``value`` is other than 2 or 5
        """
        pair_fric = self.get_pair_friction("box", "table")
        if value.shape[0] == 2:
            # Only set linear friction
            pair_fric[:2] = value
        elif value.shape[0] == 3:
            # linear friction + torsional
            pair_fric[:3] = value
        elif value.shape[0] == 5:
            # Set all 5 friction components
            pair_fric[:] = value
        else:
            raise ValueError("Friction should be a vector of 2 or 5 elements.")

    @randomization_setter("box_com")
    def set_box_com(self, value: List[float]):
        """Sets the center of mass of the hockey puck
            :param value: x,y,z list of new CoM offset w.r.t.
                          to geometric center
                          Value can also be [x,y] o [y].
        """
        assert isinstance(value, list) or isinstance(value, np.ndarray)

        if len(value) == 2:
            # Set com along z-axis to 0.0 (center)
            value = [value[0], value[1], 0.0]
        elif len(value) == 1:
            # Set comx = comz = 0.0, value is comy
            value = [0.0, value[0], 0.0]

        # self.model_args["box_com"] = list(value)
        self.model_args["box_com"] = " ".join([str(elem) for elem in value])
        self._needs_rebuilding = True

    @randomization_setter("puck_mass")
    def set_box_mass(self, new_mass):
        """
        :description: Sets the mass of the hockey puck
        :param mass: The new mass (float)
        """
        puck_mass, puck_inertia = self.get_body_mass_inertia("box")
        puck_inertia_base = puck_inertia/puck_mass
        new_inertia = puck_inertia_base * new_mass
        self.set_body_mass_inertia("box", new_mass, new_inertia)

    def set_joint_dampings(self, dampings):
        self.sim.model.dof_damping[:7] = dampings[:]

    def get_task(self):
        return self._current_task

    def set_task(self, *task):
        if len(task) != len(self.dyn_ind_to_name.values()):
            raise ValueError(f"The task given is not compatible with the dyn type selected. dyn_type:{self.dyn_type} - task:{task}")
        
        task = np.array(task)
        self._current_task = task

        if self.dyn_type == 'mf':
            self.set_box_mass(self,task[0])
            self.set_puck_friction(self, task[1:3])

        elif self.dyn_type == 'mft':
            self.set_box_mass(self, task[0])
            self.set_puck_friction(self, task[1:4])

        elif self.dyn_type == 'mfcom':
            self.set_box_com(self, task[3:5])
            if self._needs_rebuilding:
                self._rebuild_model(self)
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            self.set_box_mass(self, task[0])
            self.set_puck_friction(self, task[1:3])

        elif self.dyn_type == 'com':
            self.set_box_com(self, task[:])
            if self._needs_rebuilding:
                self._rebuild_model(self)
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden

        elif self.dyn_type == 'comy':
            self.set_box_com(self, task[:])
            if self._needs_rebuilding:
                self._rebuild_model(self)
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden

        elif self.dyn_type == 'mftcom':
            self.set_box_com(self, task[4:6])
            if self._needs_rebuilding:
                self._rebuild_model(self)
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            self.set_box_mass(self, task[0])
            self.set_puck_friction(self, task[1:4])

        elif self.dyn_type == 'mfcomd':
            self.set_box_com(self, task[3:5])
            if self._needs_rebuilding:
                self._rebuild_model(self)
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            self.set_box_mass(self, task[0])
            self.set_puck_friction(self, task[1:3])
            self.set_joint_dampings(task[5:12])

        elif self.dyn_type == 'd':
            self.set_joint_dampings(task[:7])

        else:
            raise NotImplementedError(f"Current randomization type is not implemented (3): {self.dyn_type}")

        return


    def set_random_dynamics_type(self, dyn_type='mf'):
        """Selects which dynamics to be randomized
        with the corresponding name encoding
        """
        if dyn_type == 'mf':  # mass + friction
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony'}
        elif dyn_type == 'mft':  # mass + friction + torsional friction
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'frictiont'}
        elif dyn_type == 'mfcom':  # mass + friction + CoM
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'comx', 4: 'comy'}
        elif dyn_type == 'com':  # CoM
            self.dyn_ind_to_name = {0: 'comx', 1: 'comy'}
        elif dyn_type == 'comy':  # CoM
            self.dyn_ind_to_name = {0: 'comy'}
        elif dyn_type == 'mftcom':  # mass + friction + torsional + CoM
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'frictiont', 4: 'comx', 5: 'comy'}
        elif dyn_type == 'mfcomd':  # + joint dampings
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'comx', 4: 'comy', 5: 'damping0', 6: 'damping1', 7: 'damping2', 8: 'damping3', 9: 'damping4', 10: 'damping5', 11: 'damping6'}
        elif dyn_type == 'd':  # joint dampings
            self.dyn_ind_to_name = {0: 'damping0', 1: 'damping1', 2: 'damping2', 3: 'damping3', 4: 'damping4', 5: 'damping5', 6: 'damping6'}
        else:
            raise NotImplementedError(f"Randomization dyn_type not implemented: {dyn_type}")

        self.dyn_type = dyn_type

        self._current_task = np.array(self.get_default_task())

        self.original_task = np.copy(self.get_task())
        self.task_dim = self.get_task().shape[0]
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)
        return


    def get_task_lower_bound(self, index):
        """Returns lowest possible feasible value for each dynamics"""
        lowest_value = {
                    'mass': 0.02, # 20gr
                    'frictionx': 0.1,
                    'frictiony': 0.1,
                    'frictiont': 0.001,
                    'comx': -0.05,
                    'comy': -0.05,
                    'damping0': 0,
                    'damping1': 0,
                    'damping2': 0,
                    'damping3': 0,
                    'damping4': 0,
                    'damping5': 0,
                    'damping6': 0
        }
        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task_upper_bound(self, index):
        """Returns highest possible feasible value for each dynamics"""
        highest_value = {
                    'mass': 2.0, #2kg
                    'frictionx': 3.,
                    'frictiony': 3.,
                    'frictiont': 1,
                    'comx': 0.05,
                    'comy': 0.05,
                    'damping0': 5000,
                    'damping1': 5000,
                    'damping2': 5000,
                    'damping3': 5000,
                    'damping4': 5000,
                    'damping5': 5000,
                    'damping6': 5000
        }
        return highest_value[self.dyn_ind_to_name[index]]


    def get_default_task(self):
        default_values = {
            'mass': 0.8,
            'frictionx': 0.8,
            'frictiony': 0.8,
            'frictiont': 0.01,
            'comx': 0.0,
            'comy': 0.0,
            'damping0': 100,
            'damping1': 100,
            'damping2': 100,
            'damping3': 100,
            'damping4': 100,
            'damping5': 10,
            'damping6': 0.4
        }

        default_task = [default_values[dyn] for dyn in self.dyn_ind_to_name.values()]
        return default_task

    
    ### DROPO-specific method
    def get_search_bounds_mean(self, index):
        """Get search bounds for the MEAN of the parameters optimized,
        the variance search bounds is set accordingly
        """
        search_bounds_mean = {
               'mass': (0.08, 2.0),
               'frictionx': (0.2, 2.0),
               'frictiony': (0.2, 2.0),
               'frictiont': (0.001, 0.5),
               'solref0': (0.001, 0.02),
               'solref1': (0.4, 1.),
               'comx': (-0.05, 0.05),
               'comy': (-0.05, 0.05),
               'damping0': (0,4000),
               'damping1': (0,4000),
               'damping2': (0,4000),
               'damping3': (0,4000),
               'damping4': (0,4000),
               'damping5': (0,2000),
               'damping6': (0, 200)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    ### DROPO-specific method
    def get_mjstate(self, state):
        mjstate = super().get_mjstate(state)

        # Extract all data from the state vector
        puck_pos = state[14:17]
        puck_velp = state[17:20]
        puck_orientation = state[20:23]
        puck_velr = state[23:26]

        # Put it in mjstate at the right coords
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        mjstate.qpos[puck_joint_id:puck_joint_id+3] = puck_pos
        # Convert the orientation to quaternion
        r = Rotation.from_euler(random_envs.mujoco_panda.EULER_ORDER, puck_orientation)
        mjstate.qpos[puck_joint_id+3:puck_joint_id+7] = r.as_quat()
        mjstate.qvel[puck_joint_id:puck_joint_id+3] = puck_velp
        mjstate.qvel[puck_joint_id+3:puck_joint_id+6] = puck_velr

        return mjstate


    ### DROPO-specific method
    def get_full_mjstate(self, state, template=None):

        # mjstate = super().get_mjstate(state, template) # commented out for letting the arm motion replayed without resetting
        if template is None:
            mjstate = deepcopy(self.get_sim_state())
        else:
            mjstate = deepcopy(template)

        # puck_posxy = state[14:16]
        # puck_velpxy = state[16:18]

        puck_posxyz = state[14:17]
        puck_velpxyz = state[17:20]

        # Put it in mjstate at the right coords
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        # mjstate.qpos[puck_joint_id:puck_joint_id+2] = puck_posxy
        # mjstate.qvel[puck_joint_id:puck_joint_id+2] = puck_velpxy
        mjstate.qpos[puck_joint_id:puck_joint_id+3] = puck_posxyz
        mjstate.qvel[puck_joint_id:puck_joint_id+3] = puck_velpxyz

        # mjstate.qpos[puck_joint_id+3:puck_joint_id+7] = state[18:22]
        mjstate.qpos[puck_joint_id+3:puck_joint_id+7] = state[20:24]
        mjstate.qvel[puck_joint_id+3:puck_joint_id+6] = state[24:27]
        # mjstate.qvel[puck_joint_id+3:puck_joint_id+6] = state[22:25]

        return mjstate


    ### DROPO-specific method
    def set_initial_joint_posvel(self, state, template=None):
        mjstate = super().get_mjstate(state, template)

        return mjstate

    
    ### DROPO-specific method
    def set_initial_mjstate(self, state, template=None):
        """Sets position x,y + velocity x,y + orientation"""
        mjstate = super().get_mjstate(state, template)

        # puck_posxy = state[14:16]
        # puck_velpxy = state[16:18]

        puck_posxyz = state[14:17]
        puck_velpxyz = state[17:20]

        # Put it in mjstate at the right coords
        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        # mjstate.qpos[puck_joint_id:puck_joint_id+2] = puck_posxy
        # mjstate.qvel[puck_joint_id:puck_joint_id+2] = puck_velpxy
        mjstate.qpos[puck_joint_id:puck_joint_id+3] = puck_posxyz
        mjstate.qvel[puck_joint_id:puck_joint_id+3] = puck_velpxyz

        # mjstate.qpos[puck_joint_id+3:puck_joint_id+7] = state[18:22]
        mjstate.qpos[puck_joint_id+3:puck_joint_id+7] = state[20:24]
        mjstate.qvel[puck_joint_id+3:puck_joint_id+6] = state[24:27]

        return mjstate

    ### Unused
    def set_box_pos(self, state):
        mjstate = deepcopy(self.get_sim_state())
        puck_posxy = state[14:16]

        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        mjstate.qpos[puck_joint_id:puck_joint_id+2] = puck_posxy

        return mjstate

    ### Unused
    def set_box_pos_and_rot(self, state):
        mjstate = deepcopy(self.get_sim_state())
        puck_posxy = state[14:16]

        puck_joint_id = self.sim.model.joint_name2id("puck_joint")
        mjstate.qpos[puck_joint_id:puck_joint_id+2] = puck_posxy
        mjstate.qpos[puck_joint_id+3:puck_joint_id+7] = state[18:22]

        return mjstate



"""


    Gym-registered panda push environments


"""

### Goal distribution
# push_goal_low = np.array([0.3, -0.3])
# push_goal_high = np.array([0.6, 0.3])

### Starting box distribution (see default values in class: init_box_low, init_box_high)
# init_box_low = np.array([0.4, -0.2])
# init_box_high = np.array([0.6, 0.2])

panda_start_jpos = np.array([0, 0.15, 0, -2.60, 0, 1.20, 0])

## Fixed goals for training
fixed_push_goal_a = np.array([0.75, 0.0])
fixed_push_goal_b = np.array([0.7, 0.1])
fixed_push_goal_c = np.array([0.7, -0.1])
fixed_push_goal_d = np.array([0.65, -0.2])
fixed_push_goal_e = np.array([0.7, 0.05])

### Goal A
register_panda_env(
        id="PandaPush-PosCtrl-GoalA-v0",
        entry_point="%s:PandaPushEnv" % __name__,
        model_file="franka_box.xml",
        controller=JointPositionController,
        controller_kwargs = {"clip_acceleration": False},
        action_interpolator=QuadraticInterpolator,
        action_repeat_kwargs={"start_pos": env_field("joint_pos"),
                            "start_vel": env_field("joint_vel"),
                            "dt": env_field("sim_dt")},
        model_args = {"actuator_type": "torque",
                      "with_goal": True,
                      "finger_type": "3dprinted",
                      "reduce_damping": True,
                      "limit_ctrl": False,
                      "limit_force": False,
                      "init_joint_pos": panda_start_jpos},
        max_episode_steps=500,
        env_kwargs = {"command_type": "acc",
                      "limit_power": 4,
                      "contact_penalties": True,
                      "task_reward": "target",
                      "goal_low": fixed_push_goal_a,
                      "goal_high": fixed_push_goal_a,
                      "init_jpos_jitter": 0.,
            }
        )

randomized_dynamics = ['mf', 'mft', 'mfcom', 'com', 'comy', 'mftcom', 'mfcomd', 'd']
for dyn_type in randomized_dynamics:
    register_panda_env(
            id=f"PandaPush-PosCtrl-GoalA-{dyn_type}-v0",
            entry_point="%s:PandaPushEnv" % __name__,
            model_file="franka_box.xml",
            controller=JointPositionController,
            controller_kwargs = {"clip_acceleration": False},
            action_interpolator=QuadraticInterpolator,
            action_repeat_kwargs={"start_pos": env_field("joint_pos"),
                                "start_vel": env_field("joint_vel"),
                                "dt": env_field("sim_dt")},
            model_args = {"actuator_type": "torque",
                          "with_goal": True,
                          "finger_type": "3dprinted",
                          "reduce_damping": True,
                          "limit_ctrl": False,
                          "limit_force": False,
                          "init_joint_pos": panda_start_jpos},
            max_episode_steps=500,
            env_kwargs = {"command_type": "acc",
                          "limit_power": 4,
                          "contact_penalties": True,
                          "task_reward": "target",
                          "goal_low": fixed_push_goal_a,
                          "goal_high": fixed_push_goal_a,
                          "init_jpos_jitter": 0.,
                          "randomized_dynamics": dyn_type
                }
            )


register_panda_env(
        id="PandaPush-TorqueCtrl-GoalA-v0",
        entry_point="%s:PandaPushEnv" % __name__,
        model_file="franka_box.xml",
        controller=TorqueController,
        controller_kwargs={"gravity_compensation": True},
        action_interpolator=Repeater,
        model_args = {"actuator_type": "torque", "with_goal": True,
            "finger_type": "3dprinted",
            "init_joint_pos": panda_start_jpos},
        max_episode_steps=500,
        env_kwargs = {"command_type": "torque", "limit_power": 4,
            "contact_penalties": True, "task_reward": "target",
            "goal_low": fixed_push_goal_a, "goal_high": fixed_push_goal_a,
            "init_jpos_jitter": 0.,
            }
        )

register_panda_env(
        id="PandaPush-TorqueCtrl-DropoTorques-GoalA-v0",
        entry_point="%s:PandaPushEnv" % __name__,
        model_file="franka_box.xml",
        controller=TorqueController,
        controller_kwargs={"gravity_compensation": False},
        action_interpolator=Repeater,
        model_args = {"actuator_type": "torque", "with_goal": True,
            "finger_type": "3dprinted",
            "init_joint_pos": np.array([0, 0.20, 0, -2.65, 0, 1.25, 0])},
        max_episode_steps=500,
        env_kwargs = {"command_type": "torque", "limit_power": 4,
            "contact_penalties": True, "task_reward": "target",
            "goal_low": fixed_push_goal_a, "goal_high": fixed_push_goal_a,
            "init_jpos_jitter": 0.,
            }
        )


### Goal B
register_panda_env(
        id="PandaPush-PosCtrl-GoalB-v0",
        entry_point="%s:PandaPushEnv" % __name__,
        model_file="franka_box.xml",
        controller=JointPositionController,
        controller_kwargs = {"clip_acceleration": False},
        action_interpolator=QuadraticInterpolator,
        action_repeat_kwargs={"start_pos": env_field("joint_pos"),
            "start_vel": env_field("joint_vel"),
            "dt": env_field("sim_dt")},
        model_args = {"actuator_type": "torque", "with_goal": True,
            "finger_type": "3dprinted", "reduce_damping": True,
            "limit_ctrl": False, "limit_force": False,
            "init_joint_pos": panda_start_jpos},
        max_episode_steps=500,
        env_kwargs = {"command_type": "acc", "limit_power": 4,
            "contact_penalties": True, "task_reward": "target",
            "goal_low": fixed_push_goal_b, "goal_high": fixed_push_goal_b,
            "init_jpos_jitter": 0.,
            }
        )

register_panda_env(
        id="PandaPush-TorqueCtrl-GoalB-v0",
        entry_point="%s:PandaPushEnv" % __name__,
        model_file="franka_box.xml",
        controller=TorqueController,
        controller_kwargs={"gravity_compensation": True},
        action_interpolator=Repeater,
        model_args = {"actuator_type": "torque", "with_goal": True,
            "finger_type": "3dprinted",
            "init_joint_pos": panda_start_jpos},
        max_episode_steps=500,
        env_kwargs = {"command_type": "torque", "limit_power": 4,
            "contact_penalties": True, "task_reward": "target",
            "goal_low": fixed_push_goal_b, "goal_high": fixed_push_goal_b,
            "init_jpos_jitter": 0.,
            }
        )


### Goal C
register_panda_env(
        id="PandaPush-PosCtrl-GoalC-v0",
        entry_point="%s:PandaPushEnv" % __name__,
        model_file="franka_box.xml",
        controller=JointPositionController,
        controller_kwargs = {"clip_acceleration": False},
        action_interpolator=QuadraticInterpolator,
        action_repeat_kwargs={"start_pos": env_field("joint_pos"),
            "start_vel": env_field("joint_vel"),
            "dt": env_field("sim_dt")},
        model_args = {"actuator_type": "torque", "with_goal": True,
            "finger_type": "3dprinted", "reduce_damping": True,
            "limit_ctrl": False, "limit_force": False,
            "init_joint_pos": panda_start_jpos},
        max_episode_steps=500,
        env_kwargs = {"command_type": "acc", "limit_power": 4,
            "contact_penalties": True, "task_reward": "target",
            "goal_low": fixed_push_goal_c, "goal_high": fixed_push_goal_c,
            "init_jpos_jitter": 0.,
            }
        )



# # New envs, v2 or whatever
# fixed_push_goal = {}
# fixed_push_goal["A"] = fixed_push_goal_a
# fixed_push_goal["B"] = fixed_push_goal_b
# fixed_push_goal["C"] = fixed_push_goal_c
# fixed_push_goal["D"] = fixed_push_goal_d
# fixed_push_goal["E"] = fixed_push_goal_e

# for goal in ["A", "B", "C", "D", "E"]:
#     for pen in ["0.1", "1", "3", "10", "20"]:
#         for orientation in ["none"]:
#             register_panda_env(
#                     id=f"PandaPush-PosCtrl-Goal{goal}-Pen{pen}-v0",
#                     entry_point="%s:PandaPushEnv" % __name__,
#                     model_file="franka_box.xml",
#                     controller=JointPositionController,
#                     controller_kwargs = {"clip_acceleration": False},
#                     action_interpolator=QuadraticInterpolator,
#                     action_repeat_kwargs={"start_pos": env_field("joint_pos"),
#                         "start_vel": env_field("joint_vel"),
#                         "dt": env_field("sim_dt")},
#                     model_args = {"actuator_type": "torque", "with_goal": True,
#                         "finger_type": "3dprinted", "reduce_damping": True,
#                         "limit_ctrl": False, "limit_force": False,
#                         "init_joint_pos": panda_start_jpos},
#                     max_episode_steps=500,
#                     env_kwargs = {"command_type": "acc", "limit_power": 4,
#                         "contact_penalties": True, "task_reward": "target",
#                         "goal_low": fixed_push_goal[goal], "goal_high": fixed_push_goal[goal],
#                         "init_jpos_jitter": 0., "rotation_in_obs": orientation,
#                         "control_penalty_coeff": float(pen),
#                         }
#                     )

#             register_panda_env(
#                     id=f"PandaPush-PosCtrl-Goal{goal}-Pen{pen}-NoConPen-v0",
#                     entry_point="%s:PandaPushEnv" % __name__,
#                     model_file="franka_box.xml",
#                     controller=JointPositionController,
#                     controller_kwargs = {"clip_acceleration": False},
#                     action_interpolator=QuadraticInterpolator,
#                     action_repeat_kwargs={"start_pos": env_field("joint_pos"),
#                         "start_vel": env_field("joint_vel"),
#                         "dt": env_field("sim_dt")},
#                     model_args = {"actuator_type": "torque", "with_goal": True,
#                         "finger_type": "3dprinted", "reduce_damping": True,
#                         "limit_ctrl": False, "limit_force": False,
#                         "init_joint_pos": panda_start_jpos},
#                     max_episode_steps=500,
#                     env_kwargs = {"command_type": "acc", "limit_power": 4,
#                         "contact_penalties": False, "task_reward": "target",
#                         "goal_low": fixed_push_goal[goal], "goal_high": fixed_push_goal[goal],
#                         "init_jpos_jitter": 0., "rotation_in_obs": orientation,
#                         "control_penalty_coeff": float(pen),
#                         }
#                     )


