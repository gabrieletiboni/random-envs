import pdb

import numpy as np
from mujoco_py import functions


class Controller():
    """
    The base class for all controllers
    """
    def __init__(self, sim):
        """
        :param sim: the environment to control
        """
        self.sim = sim

    def get_control(self, command):
        """
        :description: The base function for calculating the control signal given
            command. This base class simply acts as a 'passthrough'
        :param command: the command signal
        :return: the control signal
        """
        return command

    def reset(self):
        """
        :description: the reset function is called at the end of each episode.
            Reset, e.g., your integral terms and previous values for derivative
            calculation here.
        """
        return


class TorqueController(Controller):
    """
    The base class for all torque controllers.
    """
    def __init__(self, sim, gravity_compensation=False):
        """
        :param sim: the environment to control
        :param gravity_compensation: whether to include compensation terms
            in the controller
        """
        super().__init__(sim)
        self.add_grav_comp = gravity_compensation

    def get_control(self, command):
        """
        :description: The base function for calculating the control signal given command.
            This base class simply acts as a 'passthrough' and optionally adds
            the compensation terms.
        :param command: the command signal
        :return: the control signal
        """
        if self.add_grav_comp:
            return command + self.gravity_compensation()
        else:
            return command

    def gravity_compensation(self):
        """
        :description: Calculate compensation terms
        :return: compensation terms
        """
        return self.sim.sim.data.qfrc_bias[self.sim.arm_joint_index]


class PIDController(Controller):
    """
    Implements a PID controller
    """
    def __init__(self, sim, kp, ki, kd):
        """
        :param sim: the environment to control
        :param kp: proportional term gain
        :param ki: integral term gain
        :param kd: derivative term gain
        """
        super().__init__(sim)
        self.last_error = 0
        self.error_integral = 0

    def pid_control(self, error):
        dt = self.sim.model.opt.timestep
        p_term = self.kp * error
        i_term = self.ki * self.error_integral
        d_term = (error - self.last_error)/dt

        self.error_integral += dt * error
        self.last_error = error

        return p_term + i_term + d_term

    def reset(self):
        """
        :description: Reset the PID controller (zero out the accumulated error
            integral and last_error)
        """
        super().reset()
        self.last_error = 0
        self.error_integral = 0


class JointPosPID(PIDController, TorqueController):
    """
    Base class for the joint PID controllers  with torque output.
    """
    def __init__(self, sim, kp=1, ki=0, kd=0, gravity_compensation=True):
        super().__init__(sim, kp, ki, kd)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.add_grav_comp = gravity_compensation

    def get_control(self, command):
        current_pos = self.sim.joint_pos
        error = command - self.sim.joint_pos
        control = self.pid_control(error)
        if self.add_grav_comp:
            control += self.gravity_compensation()
        return control


class CartPosPID(PIDController, TorqueController):
    def __init__(self, sim, site, kp=1, ki=0, kd=0, gravity_compensation=True):
        super().__init__(sim, kp, ki, kd)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.site = site
        self.add_grav_comp = gravity_compensation

    def get_control(self, command):
        target_xyz = command[:3]
        target_rot = command[3:]
        site_pos = self.current_pos
        site_rot = np.empty(4)
        site_rot_mat = self.sim.sim.data.get_site_xmat(self.site).ravel()
        functions.mju_mat2Quat(site_rot, site_rot_mat)

        xyz_diff = target_xyz - site_pos
        ang_vel = np.ndarray(3)
        functions.mju_subQuat(ang_vel, site_rot, target_rot)

        # Now, for whatever reason, we need to flip the x error...
        ang_vel[0] = -ang_vel[0]

        # Stack the errors and push them through PID controller
        error = np.hstack([xyz_diff, ang_vel])
        out = self.pid_control(error)

        # Compute required torques using positional and rotational Jacobians
        torques_cartesian = np.dot(self.jac_pos, out[:3])
        torques_euler = np.dot(self.jac_rot, out[3:6])
        control = torques_cartesian + torques_euler

        if self.add_grav_comp:
            control += self.gravity_compensation()
        return control

    @property
    def current_pos(self):
        site_pos = self.sim.sim.data.get_site_xpos(self.site)
        return site_pos

    @property
    def jac_rot(self):
        """Jacobian for rotational angles, shaped (3, 7)."""
        J = self.sim.sim.data.get_site_jacr(self.site)
        J = J.reshape(3, -1)[:, self.sim.arm_joint_index].T
        return J

    @property
    def jac_pos(self):
        """Jacobian for positional coordinates, shaped (3, 7)."""
        J = self.sim.sim.data.get_site_jacp(self.site)
        J = J.reshape(3, -1)[:, self.sim.arm_joint_index].T
        return J


class CartesianImpedanceController(CartPosPID):
    def __init__(self, sim, site, kp=1, kd=0, gravity_compensation=True):
        super().__init__(sim, site, kp, 0, kd, gravity_compensation)


class JointImpedanceController(JointPosPID):
    def __init__(self, sim, kp=1, kd=0, gravity_compensation=True):
        super().__init__(sim, kp, 0, kd, gravity_compensation)


class JointPositionController(PIDController, TorqueController):
    """
    A pretty accurate joint position controller. Can be used when the policy is
    to be deployed with the position control interface on the real robot.

    This controller uses a feedforward term and a PID control term for corrections
    """
    def __init__(self, sim, clip_position=True, clip_acceleration=True):
        """
        :param sim: the environment with the robot to control
        :param clip_position: whether to clip positions to joint limits
        :param clip_acceleration: whether to clip accelerations to robot limits
        """
        kp = np.array([1e3, 1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
        ki = np.array([1e4, 1e4, 1e4, 1e4, 1e3, 1e3, 1e3])
        kd = np.array([1e3, 1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
        # kp = np.array([600, 600, 600, 600, 250, 150, 50])
        # ki = np.array([0, 0, 0, 0, 0, 0, 0])
        # kd = np.array([30, 30, 30, 30, 10, 10, 5])
        super().__init__(sim, kp, ki, kd)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.clip_position = clip_position
        self.clip_acceleration = clip_acceleration

    def get_control(self, command):
        if self.clip_position:
            margin = 3 * np.pi / 180
            low = self.sim.joint_qpos_min + margin
            high = self.sim.joint_qpos_max - margin
            command = np.clip(command, low, high)
        current_pos = self.sim.joint_pos

        # Get the feedforward term
        des_acc = 2*(command-self.sim.joint_pos-self.sim.dt*self.sim.joint_vel)/self.sim.dt**2
        m_q = self.get_m_q()
        feedforward = m_q @ des_acc
        # feedforward = 0

        # Get the PID control term
        error = command - self.sim.joint_pos
        pid_term = self.pid_control(error)

        # The control torque is the sum of those terms
        control = feedforward + pid_term

        if self.clip_acceleration:
            # Project torque to accelerations using the inverse of M_q
            ctrl_acc = np.linalg.inv(m_q) @ control

            # Clamp the resulting accelerations to robot limits
            acc_clamped = np.clip(ctrl_acc, -.99*self.sim.joint_qacc_max, .99*self.sim.joint_qacc_max)

            # Project clamped accelerations to torques with M_q
            control = m_q @ acc_clamped

        # Add the compensation terms
        return control + self.gravity_compensation()

    def get_m_q(self):
        """
        :return: the mass-inertia matrix of the robot (M_q)
        """
        model = self.sim.model
        data = self.sim.sim.data
        model_nv = model.nv
        full_m_buffer = np.ndarray((model_nv*model_nv))
        functions.mj_fullM(model, full_m_buffer, data.qM)
        full_m = full_m_buffer.reshape(model_nv, model_nv)[0:7, 0:7]
        return full_m



class FFJointPositionController(TorqueController):
    """
    Joint position controller with feedforward-term with fixed given desired acceleration.

    This controller uses a feedforward term with fixed given desired acceleration
    and a PD control term for corrections. This controller can also accept a target
    desired velocity to track, it doesn't just accept track steady states.
    """
    def __init__(self, sim, clip_position=True, clip_acceleration=True):
        """
        :param sim: the environment with the robot to control
        :param clip_position: whether to clip positions to joint limits
        :param clip_acceleration: whether to clip accelerations to robot limits
        """
        super().__init__(sim)
        kp = np.array([1e3, 1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
        # ki = np.array([1e4, 1e4, 1e4, 1e4, 1e3, 1e3, 1e3])
        kd = np.array([1e3, 1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
        # kp = np.array([600, 600, 600, 600, 250, 150, 50]) * 0.25
        # kd = np.array([50, 50, 50, 20, 20, 20, 10]) * 0.25
        self.kp, self.kd = kp, kd
        self.clip_position = clip_position
        self.clip_acceleration = clip_acceleration

        self.pos_errors = []
        self.vel_errors = []
        self.des_pos = []
        self.des_vel = []
        self.des_acc = []
        self.j_pos = []
        self.j_vel = []
        self.ctrl_ff_term = []
        self.ctrl_pd_term = []

    def plot_transient(self):
        import matplotlib.pyplot as plt

        self.pos_errors = np.array(self.pos_errors)
        self.vel_errors = np.array(self.vel_errors)
        self.des_pos = np.array(self.des_pos)
        self.des_vel = np.array(self.des_vel)
        self.des_acc = np.array(self.des_acc)
        self.j_pos = np.array(self.j_pos)
        self.j_vel = np.array(self.j_vel)
        self.ctrl_ff_term = np.array(self.ctrl_ff_term)
        self.ctrl_pd_term = np.array(self.ctrl_pd_term)

        fig, ax = plt.subplots(nrows=7, ncols=7)
        joints = list(range(7))
        for joint in joints:
            ax[0, joint].plot(np.arange(self.j_pos.shape[0]), self.j_pos[:, joint], c='blue', linestyle='-', marker='*', label='j_pos')
            ax[0, joint].plot(np.arange(self.des_pos.shape[0]), self.des_pos[:, joint], c='red', linestyle='--', label='des j_pos')
            ax[0, joint].legend()

            ax[1, joint].plot(np.arange(self.j_vel.shape[0]), self.j_vel[:, joint], c='blue', linestyle='-', marker='*', label='j_vel')
            ax[1, joint].plot(np.arange(self.des_vel.shape[0]), self.des_vel[:, joint], c='red', linestyle='--', label='des j_vel')
            ax[1, joint].legend()

            ax[2, joint].plot(np.arange(self.pos_errors.shape[0]), self.pos_errors[:, joint], c='blue', linestyle='-', marker='*', label='pos error')
            ax[2, joint].legend()
            ax[2, joint].set_yscale("log") 

            ax[3, joint].plot(np.arange(self.vel_errors.shape[0]), self.vel_errors[:, joint], c='blue', linestyle='-', marker='*', label='vel error')
            ax[3, joint].legend()
            ax[3, joint].set_yscale("log") 

            ax[4, joint].plot(np.arange(self.ctrl_ff_term.shape[0]), self.ctrl_ff_term[:, joint], c='green', linestyle='-', marker='*', label='ctrl_ff_term')
            ax[4, joint].legend()

            ax[5, joint].plot(np.arange(self.ctrl_pd_term.shape[0]), self.ctrl_pd_term[:, joint], c='green', linestyle='-', marker='*', label='ctrl_pd_term')
            ax[5, joint].legend()

            ax[6, joint].plot(np.arange(self.des_acc.shape[0]), self.des_acc[:, joint], c='blue', linestyle='-', marker='*', label='des_acc')
            ax[6, joint].legend()


        plt.show()

        self.pos_errors = []
        self.vel_errors = []
        self.des_pos = []
        self.des_vel = []
        self.des_acc = []
        self.j_pos = []
        self.j_vel = []
        self.ctrl_ff_term = []
        self.ctrl_pd_term = []


    def get_control(self, command):
        """
            command: (des_jpos, des_jvel, des_jacc)
        """
        des_jpos, des_jvel, des_jacc = command
        # des_jvel = 0
        # if not np.all(np.isclose(des_jvel, 0)):
        #     pdb.set_trace()

        if self.clip_position:
            margin = 3 * np.pi / 180
            low = self.sim.joint_qpos_min + margin
            high = self.sim.joint_qpos_max - margin
            des_jpos = np.clip(des_jpos, low, high)
        
        current_pos = self.sim.joint_pos
        current_vel = self.sim.joint_vel # + np.random.randn(7)*0.0011

        # Get the feedforward term
        m_q = self.get_m_q()
        feedforward = m_q @ des_jacc

        # Get the PID control term
        pid_term = (des_jpos - current_pos) * self.kp + (des_jvel - current_vel) * self.kd

        # print('pos error:', des_jpos - current_pos)
        # print('vel error:', des_jvel - current_vel)

        self.pos_errors.append(des_jpos - current_pos)
        self.vel_errors.append(des_jvel - current_vel)
        self.des_pos.append(des_jpos)
        self.des_vel.append(des_jvel)
        self.des_acc.append(des_jacc)
        self.j_pos.append(current_pos)
        self.j_vel.append(current_vel)

        # The control torque is the sum of those terms
        control = feedforward + pid_term

        # pdb.set_trace()

        self.ctrl_ff_term.append(feedforward)
        self.ctrl_pd_term.append(pid_term)

        if self.clip_acceleration:
            # Project torque to accelerations using the inverse of M_q
            ctrl_acc = np.linalg.inv(m_q) @ control

            # Clamp the resulting accelerations to robot limits
            acc_clamped = np.clip(ctrl_acc, -.99*self.sim.joint_qacc_max, .99*self.sim.joint_qacc_max)

            # Project clamped accelerations to torques with M_q
            control = m_q @ acc_clamped

        # Add the compensation terms
        return control + self.gravity_compensation()

    def get_m_q(self):
        """
        :return: the mass-inertia matrix of the robot (M_q)
        """
        model = self.sim.model
        data = self.sim.sim.data
        model_nv = model.nv
        full_m_buffer = np.ndarray((model_nv*model_nv))
        functions.mj_fullM(model, full_m_buffer, data.qM)
        full_m = full_m_buffer.reshape(model_nv, model_nv)[0:7, 0:7]
        return full_m