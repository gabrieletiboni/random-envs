import numpy as np
import gym
from random_envs.mujoco_panda.core.panda_gym_env import PandaGymEnvironment, randomization_setter
from random_envs.mujoco_panda.core.utils import register_panda_env, env_field, distance_penalty
from random_envs.mujoco_panda.core.controllers import JointImpedanceController, \
        JointPositionController, Controller, CartesianImpedanceController
from random_envs.mujoco_panda.core.interpolation import Repeater, LinearInterpolator, QuadraticInterpolator
from collections import OrderedDict


class PandaReachEnv(PandaGymEnvironment):
    def __init__(self, model_file, controller, action_interpolator, action_repeat_kwargs,
                 model_kwargs={}, controller_kwargs={}, render_camera="side_camera",
                 render_res=(320, 240), task_reward="target", command_type="new_pos",
                 acceleration_penalty_factor=1e-1, limit_power=2,
                 randomizations={}, reset_goal="random", control_penalty_coeff=1.):
        PandaGymEnvironment.__init__(self, model_file, controller, action_interpolator,
                              action_repeat_kwargs, model_kwargs, controller_kwargs,
                              render_camera, render_res, command_type,
                              acceleration_penalty_factor, limit_power,
                              randomizations)

        # Per datasheet, the reachable range is up to (around) 85.5cm
        self.goal_radius_range = .20, .80

        # Override observation space, we now also have the goal (extra 3dim)
        max_obs = np.array([np.inf]*(self.robot_obs_dim + 3))
        self.observation_space = gym.spaces.Box(-max_obs, max_obs)

        self.reset_goal = reset_goal
        self.task_reward = task_reward
        self.control_penalty_coeff = control_penalty_coeff

    def step(self, action):
        state, reward, done, info = super().step(action)
        contact_penalty = self.get_contact_penalty()
        reward += contact_penalty
        info["contact_penalty"] = contact_penalty
        return state, reward, done, info

    @randomization_setter("robot_mass")
    def set_robot_mass(self, coeff):
        self.model_args["joint_mass_scale"] = coeff
        self._needs_rebuilding = True

    def get_observation(self):
        return np.concatenate([super().get_observation(), self.goal_pos])

    @property
    def goal_distance(self):
        return np.sqrt(np.sum((self.gripper_pos - self.goal_pos)**2))

    def get_task_reward(self):
        if self.task_reward == "target":
            return -self.goal_distance
        elif self.task_reward == "goaldistpen":
            return distance_penalty(self.goal_distance, alpha=1e-3)
        elif self.task_reward == None:
            return 0
        else:
            raise ValueError(f"Unknown reward type: {self.task_reward}")

    def get_random_goal(self, sampling="cube"):
        if sampling == "cube":
            x_range = -.7, .7
            y_range = -0.4, 0.4
            z_range = .78, 1.8
            newp = np.random.uniform(np.array([x_range[0], y_range[0], z_range[0]]),
                                     np.array([x_range[1], y_range[1], z_range[1]]))
        elif sampling == "sphere":
            while True:
                # Sample a point inside a cube
                newp = np.random.uniform(-self.goal_radius_range[1], self.goal_radius_range[1], 3)
                # Is it in the right range?
                if not (self.goal_radius_range[0] < np.sqrt(np.sum(newp**2)) < self.goal_radius_range[1]):
                    continue
                # Is is above the table?
                if newp[2] < 0.:
                    continue
                break
            # Move the goal up by table height
            newp[2] += 0.8
        else:
            raise ValueError(f"Unknown goal sampling: {sampling}")
        return newp

    def set_random_goal(self, sampling="cube"):
        goal = self.get_random_goal(sampling)
        self.goal_pos = goal

    def reset(self, goal=None):
        if goal is None:
            goal = self.reset_goal

        super().reset()
        if goal == "random":
            self.set_random_goal()
        elif goal == "same":
            # Keep current goal
            pass
        elif isinstance(goal, np.ndarray):
            self.goal_pos = goal
        else:
            raise ValueError(f"Unknown goal: {goal}")

        return self.get_observation()

    def get_contact_penalty(self):
        total_contact = 0
        contacts = self.sim.data.contact
        for c in contacts:
            total_contact += c.dist**2
        return -total_contact



## Env definitions
# Basic env with impedance control, no randomizations
register_panda_env(
    id="PandaReach-ImpCtrl-v0",
    entry_point="%s:PandaReachEnv" % __name__,
    model_file="franka_table.xml",
    controller=JointImpedanceController,
    controller_kwargs={
        # Values from Panda impedance control demo
        "kp": np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]),
        "kd": np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])},
    action_interpolator=LinearInterpolator,
    action_repeat_kwargs={"start_value": env_field("joint_pos")},
    model_args = {"actuator_type": "torque", "with_goal": True,
                  "finger_type": "3dprinted"},
    max_episode_steps=500,
    env_kwargs = {"command_type": "delta_vel", "limit_power": 4}
)

# Basic env with position control, no randomizations
register_panda_env(
    id="PandaReach-PosCtrl-v0",
    entry_point="%s:PandaReachEnv" % __name__,
    model_file="franka_table.xml",
    controller=JointPositionController,
    controller_kwargs = {"clip_acceleration": False},
    action_interpolator=LinearInterpolator,
    action_repeat_kwargs={"start_value": env_field("joint_pos")},
    model_args = {"actuator_type": "torque", "with_goal": True,
                  "finger_type": "3dprinted", "damping_mode": "armature",
                  "limit_ctrl": False, "limit_force": False},
    max_episode_steps=500,
    env_kwargs = {"command_type": "delta_vel", "limit_power": 4}
)

# Basic env with the arm controlled by a MoCap body
# similar to OpenAI Fetch reach, no randomizations
register_panda_env(
    id="PandaReach-v0",
    entry_point="%s:PandaReachEnv" % __name__,
    model_file="franka_table.xml",
    controller=Controller,
    action_interpolator=Repeater,
    model_args = {"actuator_type": "mocap", "with_goal": True,
                  "finger_type": "3dprinted"},
    max_episode_steps=500,
    env_kwargs = {"command_type": None, "limit_power": 4}
)

# Test envs for position controllers
goals = {}
goals["A"] = np.array([0.5, 0, 1.6])
goals["B"] = np.array([0.7, 0, 0.9])
goals["C"] = np.array([0.4, 0.4, 1.0])

for goal in "ABC":
    for pen in [1, 10, 100]:
        # Regular target pen
        register_panda_env(
            id=f"PandaReach-PosCtrl-Goal{goal}-Pen{pen}-v0",
            entry_point="%s:PandaReachEnv" % __name__,
            model_file="franka_table.xml",
            controller=JointPositionController,
            controller_kwargs = {"clip_acceleration": False},
            action_interpolator=QuadraticInterpolator,
            action_repeat_kwargs={"start_pos": env_field("joint_pos"),
                "start_vel": env_field("joint_vel"),
                "dt": env_field("sim_dt")},
            model_args = {"actuator_type": "torque", "with_goal": True,
                "finger_type": "3dprinted", "damping_mode": "armature",
                "limit_ctrl": False, "limit_force": False},
            max_episode_steps=500,
            env_kwargs = {"command_type": "acc", "limit_power": 4,
                "reset_goal": goals[goal], "control_penalty_coeff": pen}
        )

        # new better goal pen
        register_panda_env(
            id=f"PandaReach-PosCtrl-Goal{goal}-Pen{pen}-GDP-v0",
            entry_point="%s:PandaReachEnv" % __name__,
            model_file="franka_table.xml",
            controller=JointPositionController,
            controller_kwargs = {"clip_acceleration": False},
            action_interpolator=QuadraticInterpolator,
            action_repeat_kwargs={"start_pos": env_field("joint_pos"),
                "start_vel": env_field("joint_vel"),
                "dt": env_field("sim_dt")},
            model_args = {"actuator_type": "torque", "with_goal": True,
                "finger_type": "3dprinted", "damping_mode": "armature",
                "limit_ctrl": False, "limit_force": False},
            max_episode_steps=500,
            env_kwargs = {"command_type": "acc", "limit_power": 4,
                "reset_goal": goals[goal], "control_penalty_coeff": pen,
                "task_reward": "goaldistpen"}
        )


