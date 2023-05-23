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

