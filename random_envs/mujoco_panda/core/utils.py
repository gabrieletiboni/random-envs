import pdb

from gym.envs import register
import numpy as np


def env_field(field_name):
    """
    :description: Returns a getter function for the given field of the object.
        Used mainly when instantiating an interpolator which needs to know what
        value to start interpolating from.
    :param field_name: Name of the class field
    :return: a getter function that, after receiving an object, will return a
        getter for field ``field_name`` of the received object.
    """

    def field_getter_mid(self):
        def field_getter_inner():
            return getattr(self, field_name)
        return field_getter_inner
    return field_getter_mid

def soft_tanh_limit(x, low, high, betas=(0.35, 0.35), linear_coeff=1e-3, square_coeff=0.3):
    norm_x = (x - low) / (high - low)
    alpha_min, alpha_max = norm_x, 1-norm_x
    penalty = np.zeros_like(x)
    val_low = .5 * (1 - np.tanh(1/(1-alpha_min/betas[0]) - betas[0]/alpha_min))
    val_high = .5 * (1 - np.tanh(1/(1-alpha_max/betas[1]) - betas[1]/alpha_max))
    idx_low = alpha_min < betas[0]
    idx_high = alpha_max < betas[1]
    penalty[idx_low] = val_low[idx_low]
    penalty[idx_high] = val_high[idx_high]
    penalty[alpha_min < 0] = 1
    penalty[alpha_max < 0] = 1

    linear_term = (np.fmax(0, alpha_min) + np.fmax(0, alpha_max))
    square_term = np.fmin((norm_x*2-1)**2, 1.)

    return square_term*square_coeff + penalty*(1-square_coeff) + linear_coeff * linear_term

def distance_penalty(d, w=1, v=1, alpha=1e-3):
    return -w*d**2 - v*np.log(d**2 + alpha)


def get_preprocess_action(env, command_type, clip=True):
    """Preprocess policy action in [-1,1] to the corresponding interval
        before the action interpolator and before the low-level controller.
        
        It uses the high-level policy frequency (env.dt=20Hz) to compute
        the desired target position or torque.

        normalized vel ---> target pos after 20ms
        normalized acc ---> target pos after 20ms
        normalized torque ---> real torque
        ...
    """
    if command_type == "delta_vel":
        def preprocess_action(action):
            # Assume actions are vel deltas and output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            acc = action * env.joint_qacc_max
            delta_vel = acc * env.dt
            new_vel = env.joint_vel + delta_vel
            new_vel = np.clip(new_vel, env.joint_qvel_min, env.joint_qvel_max)
            delta_pos = new_vel * env.dt
            new_pos = env.joint_pos + delta_pos
            new_pos = np.clip(new_pos, env.joint_qpos_min, env.joint_qpos_max)
            return new_pos
    elif command_type == "acc_npc":
        def preprocess_action(action):
            # Assume actions are vel deltas and output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            acc = action * env.joint_qacc_max
            delta_vel = acc * env.dt
            # Clip velocity
            end_vel = env.joint_vel + delta_vel
            true_acc = (end_vel - env.joint_vel)/env.dt
            new_pos = 0.5*true_acc*env.dt**2 + env.joint_vel*env.dt + env.joint_pos
            return new_pos
    elif command_type == "acc":
        def preprocess_action(action):
            # Assume actions are vel deltas and output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            acc = action * env.joint_qacc_max
            delta_vel = acc * env.dt
            # Clip velocity
            end_vel = env.joint_vel + delta_vel
            end_vel = np.clip(end_vel, env.joint_qvel_min, env.joint_qvel_max)
            true_acc = (end_vel - env.joint_vel)/env.dt
            new_pos = 0.5*true_acc*env.dt**2 + env.joint_vel*env.dt + env.joint_pos
            new_pos = np.clip(new_pos, env.joint_qpos_min, env.joint_qpos_max)
            return new_pos
    elif command_type == "acc-debug":
        def preprocess_action(action):
            # Assume actions are vel deltas and output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            acc = action * env.joint_qacc_max
            delta_vel = acc * env.dt
            # Clip velocity
            end_vel = env.joint_vel + delta_vel
            end_vel = np.clip(end_vel, env.joint_qvel_min, env.joint_qvel_max)
            true_acc = (end_vel - env.joint_vel)/env.dt
            
            # return env.joint_pos, env.joint_vel + np.random.randn(7)*0.0011, true_acc
            return env.joint_pos, env.joint_vel, true_acc
    elif command_type == "delta_pos":
        def preprocess_action(action):
            # Assume actions are pos deltas and output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            factor = (action+1)/2
            diff = (env.joint_qvel_max - env.joint_qvel_min)
            vel = factor * diff + env.joint_qvel_min
            delta_pos = vel * env.dt
            new_pos = env.joint_pos + delta_pos
            return new_pos
    elif command_type == "new_pos":
        def preprocess_action(action):
            # Assume actions are output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            factor = (action+1)/2
            diff = (env.joint_qpos_max - env.joint_qpos_min)
            return factor * diff + env.joint_qpos_min
    elif command_type == "halftorque":
        def preprocess_action(action):
            # Assume actions are output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            return action * env.joint_tau_max * 0.5
    elif command_type == "torque":
        def preprocess_action(action):
            # Assume actions are output of tanh...
            if clip:
                action = np.clip(action, -1, 1)
            return action * env.joint_tau_max
    elif command_type == None:
        def preprocess_action(action):
            if clip:
                action = np.clip(action, -1, 1)
            return action
    else:
        raise ValueError(f"Unknown command type: {command_type}. Available:" \
                "new_pos, delta_pos, delta_vel")
    return preprocess_action


def get_dim(param):
    """
    :description: Returns the dimension of ``param``
    :param param: the array/float value to get dimensionality of
    :return: the number of elements in ``param``
    """
    if isinstance(param, (float, int)):
        return 1
    elif isinstance(param, np.ndarray):
        return len(param)
    elif param == None:
        return 0
    else:
        raise TypeError(f"Don't know how to handle {param} of type {type(param)}")


def register_panda_env(id, entry_point, model_file, controller, action_interpolator,
                       action_repeat_kwargs={}, model_args={}, env_kwargs={},
                       render_camera="side_camera", controller_kwargs={},
                       **kwargs):
    """
    :description: Register a new Panda environment, such that it can be instantiated
        with a call to ``gym.make``.
    :param id: The name used for registration,
    :param entry_point: The Python class of the environment. It has to include
        the module name (e.g. ``reach:PandaReachEnv``; when calling from the
        same file where the class is declared, it's best to simply use
        ``"%s:ClassName" % __name__``)
    """
    register(id=id,
             entry_point=entry_point,
             kwargs={"model_file": model_file,
                     "controller": controller,
                     "action_interpolator": action_interpolator,
                     "action_repeat_kwargs": action_repeat_kwargs,
                     "model_kwargs": model_args,
                     "controller_kwargs": controller_kwargs,
                     "render_camera": render_camera,
                     **env_kwargs},
             **kwargs)
