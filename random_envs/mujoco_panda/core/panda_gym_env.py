import gym
import numpy as np
from random_envs.mujoco_panda.core.env import Environment
from random_envs.mujoco_panda.core.controllers import Controller, JointPositionController, \
        CartesianImpedanceController, JointImpedanceController, \
        TorqueController
from random_envs.mujoco_panda.core.interpolation import LinearInterpolator
from random_envs.mujoco_panda.core.utils import env_field, register_panda_env, \
    soft_tanh_limit, get_dim, get_preprocess_action
from scipy.stats import loguniform
from copy import deepcopy
import pdb


class randomization_setter():
    """
    This decorator is used to denote functions which set domain randomization
    parameter values. It maintains a global registry of all randomization
    parameters and their respective setter values.
    """
    registry = {}

    def __init__(self, arg):
        self._arg = arg
        self._f_name = None
        self._f_class = None
        self._call_func = self.call_decorator

    def call_decorator(self, f):
        """
        :description: This function is called when the function is defined.
            Later, __call__ is remapped to the actual dynamics setter through
            self._call_func
        """
        classname, fname = f.__qualname__.rsplit(".", 1)
        self._f_name = fname
        self._f_class = classname
        self._call_func = f
        if classname not in self.registry:
            self.registry[classname] = {}
        self.registry[classname][self._arg] = f
        return self

    def __call__(self, *args, **kwargs):
        return self._call_func(*args, **kwargs)


class PandaGymEnvironment(gym.Env, Environment):
    """
    The base class for all Panda gym environments.
    """
    def __init__(self, model_file, controller, action_interpolator, action_repeat_kwargs,
                 model_kwargs={}, controller_kwargs={}, render_camera="side_camera",
                 render_res=(320, 240), command_type=None,
                 acceleration_penalty_factor=1e-1, limit_power=2,
                 randomizations={}, init_pos_jitter=0.2):
        gym.Env.__init__(self)
        Environment.__init__(self, model_file, init_pos_jitter=init_pos_jitter, **model_kwargs)
        self.acceleration_penalty_factor = acceleration_penalty_factor
        self.limit_power = limit_power
        if "num" not in action_repeat_kwargs:
            # No value was provided; just fix it to provide 20ms control intervals
            desired_interval = 2e-2
            sim_dt = self.sim.model.opt.timestep
            repeat_num = desired_interval / sim_dt
            repeat_num_rounded = int(np.round(repeat_num))
            round_err = np.abs(repeat_num - repeat_num_rounded)
            if round_err > 1e-4:
                actual_dt = repeat_num_rounded * sim_dt
                print(f"Repeater has a large rounding error. Expected sim dt:"
                      f" {desired_interval}. Actual: {actual_dt}")
            action_repeat_kwargs["num"] = repeat_num_rounded

        self.control_penalty_coeff = 1.
        self._action_repeat_kwargs = dict(action_repeat_kwargs)
        self._action_repeat = action_interpolator
        self.interpolate = self._build_interpolator()
        self.controller = controller(self, **controller_kwargs)
        self.render_camera = render_camera
        self.render_res = render_res
        self.callbacks = []

        # Determine actions and observations
        # Cartesian position control with mocap
        if controller == Controller and self.mocap_control:
            # MoCap body control?
            # Observation is end-effector pos and vel
            self._robot_obs_dim = 6

            # Action is end-effector velocity
            max_action = np.ones(3)

            def preprocess_action(action):
                return action * self.max_endeff_vel
            self.robot_observation = lambda : \
                    np.concatenate([self.gripper_pos, self.gripper_vel])
            self.preprocess_action = preprocess_action

        elif controller == TorqueController:
            # Torque control
            assert command_type in (None, "torque", "halftorque")
            self._robot_obs_dim = 14

            # Action is desired joint position
            max_action = np.ones(7)
            self.robot_observation = lambda : np.concatenate([self.joint_pos, self.joint_vel])
            self.preprocess_action = get_preprocess_action(self, command_type)

        elif controller == CartesianImpedanceController:
            # Observation is end-effector pos, orientation and vel
            self._robot_obs_dim = 13

            # Action is desired position and orientation (quaternion)
            max_action = np.array([np.inf]*7)

            def preprocess_action(action):
                # Normalize the orientation quaternion
                action[3:] /= np.sqrt(np.sum(action[3:]**2))
                # Convert to double (that's what some MuJoCo functions expect)
                action = action.astype(np.float64)
                return action

            self.robot_observation = lambda : \
                np.concatenate([self.gripper_pos, self.gripper_quat,
                        self.gripper_vel, self.gripper_ang_vel])

            self.preprocess_action = preprocess_action

        elif controller == JointImpedanceController:
            # Observation is joint pos and vel
            self._robot_obs_dim = 14

            # Action is desired joint position
            max_action = np.ones(7)
            self.robot_observation = lambda : np.concatenate([self.joint_pos, self.joint_vel])
            self.preprocess_action = get_preprocess_action(self, command_type)

        elif controller == JointPositionController:
            # Observation is joint pos and vel
            self._robot_obs_dim = 14

            # Action is desired joint position
            max_action = np.ones(7)
            self.robot_observation = lambda : np.concatenate([self.joint_pos, self.joint_vel])
            self.preprocess_action = get_preprocess_action(self, command_type)

        else:
            print(f"Default action spaces and observations not defined "
                  f"for controller type {controller}. Define it in PandaGymEnv "
                  f"or manually set preprocess_action and robot_observation "
                  f"in the derived class.")
            self._robot_obs_dim = 0
            max_action = np.array([np.inf]*7)
            self.preprocess_action = lambda x: x
            self.robot_observation = lambda : np.array([])

        max_obs = np.array([np.inf]*self.robot_obs_dim)
        self.action_space = gym.spaces.Box(-max_action, max_action)
        self.observation_space = gym.spaces.Box(-max_obs, max_obs)

        # Initialize meta learning/domain randomization stuff
        self._needs_rebuilding = False
        self.randomizations = randomizations
        self.task_indices = self._get_task_variable_indices()
        self._randomization_setters = {}
        self._init_setters()
        self._task = None
        # self.set_random_task() # Handle the randomization in slide.py (temporary change by gabriele_dev)

    @property
    def robot_obs_dim(self):
        return self._robot_obs_dim

    def _init_setters(self):
        """
        This method registers domain randomizatrion callbacks defined with
        the @randomization_setter decorator. It needs to be called in the
        `__init__` method of each class that subclasses `PandaGymEnvironment`; since
        it is done inside `PandaGymEnvironment`'s own `__init__` function, it is usually
        achieved with `super().__init__()` and does not require any special attention.
        """
        for cls in type(self).mro():
            clsname = cls.__name__
            if clsname in randomization_setter.registry:
                new_ones = randomization_setter.registry[clsname]
                for name, fn in new_ones.items():
                    self._randomization_setters[name] = fn.__get__(self, cls)

    def _build_interpolator(self):
        action_repeat_kwargs = dict(self._action_repeat_kwargs)

        # Link env_field objects to this environment instance
        for k, v in action_repeat_kwargs.items():
            if hasattr(v, "__call__") and hasattr(v, "__name__") \
                    and v.__name__ == "field_getter_mid":
                action_repeat_kwargs[k] = v(self)

        interpolate = self._action_repeat(**action_repeat_kwargs)
        return interpolate

    def set_task_property(self, prop, value):
        """
        :description: This method calls the randimization setter for the given property (as
            defined by the `@randomization_setter` decorator) with the given `value`.

        :param prop:  the name of the property (same as passed to
                `@randomization_setter`)
        :param value: the new value for `prop`
        :raises ValueError:  if `prop` does not have a defined setter or is
            not randomized in the current environment
        """
        if prop not in self.randomizations:
            raise ValueError("Property {prop} is not randomized in current env!")
        if prop not in self._randomization_setters:
            raise ValueError(f"Randomization setter not defined for property {prop}!")
        self._randomization_setters[prop](value)

    @property
    def goal_pos(self):
        """
        :return: position of the goal site
        :raises: ValueError if the environment does not have a goal pos
        """
        goal_site_id = self.sim.model.site_name2id("goal")
        return self.sim.model.site_pos[goal_site_id]

    @goal_pos.setter
    def goal_pos(self, value):
        """
        :description: moves the goal site to the given position
        :param value: the target position, array of 2 or 3 elements. If dim is
            2, only the x and y positions are changed
        :raises: ValueError if the environment does not have a goal pos
        """
        num_dim = value.shape[0]
        assert num_dim in (2, 3)
        goal_site_id = self.sim.model.site_name2id("goal")
        self.sim.model.site_pos[goal_site_id][:num_dim] = value


    def reset(self):
        """
        :description: This method resets the environment to the initial state
        :raises RuntimeError: if properties randomized through template
            arguments were modified, but the environment has not been rebuild
        :returns: the result of `PandaGymEnvironment.reset()`
        """

        if self._needs_rebuilding:
            raise RuntimeError("Env needs to be rebuilt after a parameter change! " \
                    "Most likely the randomizations are not defined properly. "\
                    "You need to call _rebuild_model() after every change to a "\
                    "parameter changed inside the XML.")

        Environment.reset(self)
        self.interpolate.reset()
        for cb in self.callbacks:
            cb.post_reset(self)
        return self.get_observation()

    def render(self, mode="human"):
        if mode == "human":
            Environment.render(self)
        elif mode == "rgb_array":
            rgb = Environment.offscreen_render(self, self.render_camera,
                    self.render_res, depth=False)
            return rgb

    def get_observation(self):
        return self.robot_observation()

    def step(self, action):
        if self._needs_rebuilding:
            raise RuntimeError("Model parameters have changed! Please rebuild it and reset!")

        action = self.preprocess_action(action)
        for _target in self.interpolate(action):
            control = self.controller.get_control(_target)
            self.apply_joint_motor_command(control)
            self.sim.step()
            for cb in self.callbacks:
                cb.post_step(self, action, _target, control)

        state = self.get_observation()
        task_reward = self.get_task_reward()
        norm_acc = np.abs(self.joint_acc)/self.joint_qacc_max
        norm_vel = np.abs(self.joint_vel)/self.joint_qvel_max
        position_penalty = soft_tanh_limit(self.joint_pos, self.joint_qpos_min,
                self.joint_qpos_max, square_coeff=0., betas=(0.03, 0.03)).sum()
        velocity_penalty = soft_tanh_limit(self.joint_vel, self.joint_qvel_min,
                self.joint_qvel_max, betas=(0.2, 0.2), square_coeff=0.5).sum()
        acceleration_penalty = soft_tanh_limit(self.joint_acc,
                -self.joint_qacc_max, self.joint_qacc_max, betas=(0.2, 0.2), square_coeff=0.5).sum()
        control_penalty = velocity_penalty + acceleration_penalty + position_penalty
        control_penalty *= -self.control_penalty_coeff

        info = {"task_reward": task_reward,
                "velocity_penalty": -velocity_penalty*self.control_penalty_coeff,
                "position_penalty": -position_penalty*self.control_penalty_coeff,
                "acceleration_penalty": -acceleration_penalty*self.control_penalty_coeff,
                "velocity_over_limit": (1-self.check_joint_velocity_limit()).sum(),
                "position_over_limit": (1-self.check_joint_position_limit()).sum(),
                "acceleration_over_limit": (1-self.check_joint_acceleration_limit()).sum(),
                "control_penalty": control_penalty}
        reward = task_reward + control_penalty
        return state, reward, False, info

    def get_task_reward(self):
        return 0

    @property
    def dt(self):
        """
        :return: The time change between environment step calls, equal to the
            simulation dt time the number of inner steps (from the interpolator)
        """
        return self.sim.model.opt.timestep * self.interpolate.num

    def _get_task_variable_indices(self):
        indices = {}
        current_idx = 0
        for param, rng in self.randomizations.items():
            if param[0] == "_" and rng is None:
                continue
            param_dim1 = get_dim(rng[0])
            param_dim2 = get_dim(rng[1])
            # Make sure min and max have the same dim
            assert param_dim1 == param_dim2
            # Make sure the param hasn't been given twice
            assert param not in indices
            end_idx = current_idx + param_dim1
            indices[param] = (current_idx, end_idx)
            current_idx = end_idx
        return indices

    # This is a special key; it indicates when we need to compile the model.
    # It only makes sense when used with OrderedDict
    # Needed when params changed directly in mjmodel are randomized together
    # with params that need to be changed in the XML
    @randomization_setter("_rebuild_model")
    def _rebuild_model(self, _=None):
        self.build_model(self.model_args)
        self.interpolate = self._build_interpolator()
        self._needs_rebuilding = False

    @randomization_setter("controller_kp")
    def _set_controller_kd(self, value):
        assert hasattr(self.controller, "kp")
        self.controller.kp[:] = value

    @randomization_setter("controller_kd")
    def _set_controller_kd(self, value):
        assert hasattr(self.controller, "kd")
        self.controller.kd[:] = value

    def sample_tasks(self, num_tasks):
        return [self.sample_task() for _ in range(num_tasks)]

    def sample_task(self):
        # Gabriele has temporarily implemented DR in child class
        raise NotImplementedError('Parent sample_task() called instead of child method')

        task = []
        for param, rng in self.randomizations.items():
            if param[0] == "_" and rng is None:
                continue

            if len(rng) > 2:
                rand_type = rng[2]
            else:
                rand_type = "uniform"

            if rand_type == "uniform":
                value = np.random.uniform(rng[0], rng[1])
            elif rand_type == "loguniform":
                dist = loguniform(rng[0], rng[1])
                value= dist.rvs(get_dim(rng[0]))
            elif rand_type == "normal":
                value = np.random.normal(*rng[0].shape) * rng[1] + rng[0]
            else:
                raise ValueError(f"Unknown distribution {rand_type} for {param}")

            if isinstance(value, np.ndarray):
                task.extend(value)
            elif isinstance(value, (int, float)):
                task.append(value)
            else:
                # Shouldn't happen...
                raise TypeError("Unknown task type")

        return np.array(task)

    def set_random_task(self):
        """
        :description: This method samples a random task from the task space and
            immediately sets it in the environment.
        :returns: The new task value
        """
        # Gabriele has temporarily implemented DR in child class
        raise NotImplementedError('Parent set_random_task() called instead of child method')

        task = self.sample_tasks(1)[0]
        self.set_task(task)
        return task

    # def set_task(self, task):
    #     print('WARNING - Parent set_task called')

    #     self._task = task
    #     for param in self.randomizations:
    #         if param[0] == "_":
    #             value = None
    #         else:
    #             idx_low, idx_high = self.task_indices[param]
    #             value = task[idx_low:idx_high]
    #         self.set_task_property(param, value)

    #     if self._needs_rebuilding:
    #         self.build_model(self.model_args)

    # def get_task(self):
    #     print('WARNING - Parent get_task called')

    #     return self._task

    def randomization_setter(self, param_name):
        def inner(func):
            return func
        print('rand setter with,', param_name)
        if param_name in self._randomization_setters:
            raise ValueError("Redefining randomization setter for", param_name)
        self._randomization_setters[param_name] = inner
        return inner

    def set_sim_state(self, state):
        return self.sim.set_state(state)

    def get_sim_state(self):
        return self.sim.get_state()

    def get_mjstate(self, state, template=None):
        if template is None:
            mjstate = deepcopy(self.get_sim_state())
        else:
            mjstate = deepcopy(template)

        mjstate.qpos[:7] = state[:7]
        mjstate.qvel[:7] = state[7:14]
        return mjstate

