from random_envs.mujoco_panda.core.panda_gym_env import PandaGymEnvironment
import numpy as np
import os


class Callback():
    """
    :description: Base class for all environment callbacks
    """
    def __init__(self):
        pass

    def post_step(self, env: PandaGymEnvironment, action: np.ndarray,
                  _target: np.ndarray, control: np.ndarray) -> None:
        """
        :description: This callback will be called every time after the
            simulation is stepped (note that this refers to SIMULATION steps,
            not the environment steps; an environment step usually consists of
            multiple simulation steps). The following parameters will be passed
            to the callback by the environment:"
        :param env: the Gym environment being stepped,
        :param action: the action that was issued,
        :param _target: the target value for the controller (output of the
            interpolator)
        :param control: the control signal set to the actuators (output of the
            controller)
        """
        pass

    def post_reset(self, env: PandaGymEnvironment) -> None:
        """
        :description: This callback will be called every time the environment
            is reset.
        :param env: the Gym environment being reset,
        """
        pass


class CheckJointLimitsCallback(Callback):
    """
    :description: A callback that checks if joint positions, velocities and
        accelerations are within limits specified in the datasheet, and prints
        a warning if they aren't.
    """
    def __init__(self, check_pos=True, check_vel=True, check_acc=True):
        """
        :param check_pos: whether to check joint positions,
        :param check_vel: whether to check joint velocities,
        :param check_acc: whether to check joint accelerations,
        """
        self.check_pos = check_pos
        self.check_vel = check_vel
        self.check_acc = check_acc

    def post_step(self, env: PandaGymEnvironment, action: np.ndarray,
                  _target: np.ndarray, control: np.ndarray) -> None:
        with np.printoptions(precision=2, suppress=True):
            if self.check_pos:
                pos_limit = env.check_joint_position_limit()
                if not np.all(pos_limit):
                    qpos = env.sim.data.qpos[env.arm_joint_index]
                    where = np.where(pos_limit != 1)
                    str_where = ", ".join(map(str, list(where[0])))
                    print(" !!! Joint position limit exceeded for joints "
                          f"{str_where} ({qpos[where]} not in {env.joint_qpos_min[where]}, {env.joint_qpos_max[where]})")

            if self.check_vel:
                vel_limit = env.check_joint_velocity_limit()
                if not np.all(vel_limit):
                    qvel = env.sim.data.qvel[env.arm_joint_index]
                    where = np.where(vel_limit != 1)
                    str_where = ", ".join(map(str, list(where[0])))
                    print(" !!! Joint velocity limit exceeded for joints "
                          f"{str_where} ({qvel[where]} not in {env.joint_qvel_min[where]}, {env.joint_qvel_max[where]})")

            if self.check_acc:
                acc_limit = env.check_joint_acceleration_limit()
                if not np.all(acc_limit):
                    qacc = env.sim.data.qacc[env.arm_joint_index]
                    where = np.where(acc_limit != 1)
                    str_where = ", ".join(map(str, list(where[0])))
                    print(" !!! Joint acceleration limit exceeded for joints "
                          f"{str_where} (abs {qacc[where]} > {env.joint_qacc_max[where]})")

class SaveJointPosCallback():
    """
    :description: A callback that saves all intermediate joint positions into
        a text file. The data is dumped when the environment is reset. The
        default filename is positions_{episode}.{format}, where episode is the
        episode number and format is passed to the ``__init__`` function
        (either txt or npy).
    """
    def __init__(self, path=".", include_timestamp=True, file_format="txt"):
        """
        :param path: the path where the files will be saved. Defaults to the
            current directory.
        :param include_timestamp: whether to include simulation time as the
            first column,
        :param file_format: output file format (txt for a text file or npy for 
            a numpy array)
        """
        self.path = path
        self.include_timestamp = include_timestamp
        self.timestamps = []
        self.positions = []
        self.velocities = []
        self.actions = []
        self.targets = []
        self.control_commands = []
        self.file_id = 0
        if file_format not in ("txt", "npy", "pkl"):
            raise ValueError("Unknown file format. Use npy or txt.")
        self.format = file_format

    def post_step(self, env: PandaGymEnvironment, action: np.ndarray,
                 _target: np.ndarray, control: np.ndarray) -> None:
        self.timestamps.append(env.time)
        self.positions.append(env.joint_pos.copy())
        self.velocities.append(env.joint_vel.copy())
        self.targets.append(_target.copy())
        self.control_commands.append(control.copy())
        self.actions.append(action.copy())

    @property
    def data_frame(self):
        import pandas as pd
        timestamps = np.array(self.timestamps)
        positions = np.array(self.positions)
        targets = np.array(self.targets)
        velocities = np.array(self.velocities)
        actions = np.array(self.actions)
        control_commands = np.array(self.control_commands)

        action_dim = actions.shape[1]
        dct = {x : [] for x in ["time"] +
                [f"jpos{i}" for i in range(7)] +
                [f"jvel{i}" for i in range(7)] +
                [f"action{i}" for i in range(action_dim)] +
                [f"control{i}" for i in range(7)] +
                [f"target{i}" for i in range(7)]}
        dct["time"].extend(timestamps)
        for i in range(7):
            dct[f"jpos{i}"].extend(positions[:,  i])
            dct[f"jvel{i}"].extend(velocities[:,  i])
            dct[f"control{i}"].extend(control_commands[:,  i])
            dct[f"target{i}"].extend(targets[:,  i])
        for i in range(action_dim):
            dct[f"action{i}"].extend(actions[:,  i])
        df = pd.DataFrame(dct)
        return df

    @property
    def data_array(self):
        """
        :return: data saved so far for the current episode
        """
        timestamps = np.array(self.timestamps).reshape(-1, 1)
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        actions = np.array(self.actions)
        control_commands = np.array(self.control_commands)
        to_concat = (timestamps, positions, velocities, actions, control_commands)
        data_array = np.concatenate(to_concat, axis=-1)

        if not self.include_timestamp:
            data_array = data_array[:, 1:]
        return data_array

    @property
    def next_fname(self):
        """
        :return: the filename where the next episode will be saved
        """
        return f"positions_{self.file_id}.{self.format}"

    @property
    def last_fname(self):
        """
        :return: the filename where the previous episode was saved
        """
        return f"positions_{self.file_id-1}.{self.format}"

    def post_reset(self, env: PandaGymEnvironment):
        if len(self.positions) == 0:
            return
        fname = os.path.join(self.path, self.next_fname)
        if self.format == "npy":
            data_array = self.data_array
            np.save(fname, data_array)
        elif self.format == "txt":
            data_array = self.data_array
            np.savetxt(fname, data_array)
        elif self.format == "pkl":
            dataframe = self.data_frame
            dataframe.to_pickle(fname)
        else:
            raise ValueError("Wrong file type")
        self.timestamps = []
        self.positions = []
        self.velocities = []
        self.actions = []
        self.targets = []
        self.control_commands = []

        self.file_id += 1

