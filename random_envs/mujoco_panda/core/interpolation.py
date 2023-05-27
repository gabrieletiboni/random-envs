import pdb

import numpy as np
from scipy import interpolate


class Repeater():
    def __init__(self, num):
        self.num = num

    def __call__(self, action):
        for _ in range(self.num):
            yield action

    def reset(self):
        return


class DummyAccelerationIntegrator(Repeater):
    """
        Given an acceleration, simply returns the final
        target pos and target vel after 20ms at each low-level timestep.

        Simulates a steady-state tracker.
    """
    def __init__(self, num, dt):
        super().__init__(num)
        self._dt = dt

    def __call__(self, cmd):
        cur_pos, cur_vel, acc = cmd
        dt = self.dt
        env_dt = 0.02
        for n in range(self.num):
            # t = n * dt
            t = env_dt
            target_pos = cur_pos + t * cur_vel + 0.5 * acc * t ** 2
            target_vel = cur_vel + t * acc
            yield (target_pos, target_vel, acc)

    @property
    def dt(self):
        if callable(self._dt):
            return self._dt()
        else:
            return self._dt


class AccelerationIntegrator(Repeater):
    """
        Given a desired acceleration, compute the integrated
        jpos and jvel in a span of 20ms in the future.
    """
    def __init__(self, num, dt):
        super().__init__(num)
        self._dt = dt

    def __call__(self, cmd):
        cur_pos, cur_vel, acc = cmd
        dt = self.dt
        env_dt = 0.02
        for n in range(self.num):
            t = n * dt
            target_pos = cur_pos + t * cur_vel + 0.5 * acc * t ** 2
            target_vel = cur_vel + t * acc
            yield (target_pos, target_vel, acc)

    @property
    def dt(self):
        if callable(self._dt):
            return self._dt()
        else:
            return self._dt

class QuadraticInterpolator(Repeater):
    def __init__(self, num, start_pos, start_vel, dt, command_reference=True):
        super().__init__(num)
        self._start_pos = start_pos
        self._start_vel = start_vel
        self._dt = dt
        self.previous_command = None
        self.command_reference = command_reference

    def __call__(self, action):
        dt = self.dt
        env_dt = 0.02
        pos0 = self.start_pos.copy()
        reference = self.reference.copy()
        diff = action - reference
        vel0 = self.start_vel.copy()
        acc = 2*(diff - env_dt*vel0)/env_dt**2
        for n in range(self.num):
            t = n*dt
            target = reference + t*vel0 + 0.5*acc*t**2
            yield target
        self.previous_command = action

    @property
    def reference(self):
        if self.command_reference:
            if self.previous_command is None:
                self.previous_command = self.start_pos
            reference = self.previous_command
        else:
            reference = self.start_pos
        return reference

    @property
    def dt(self):
        if callable(self._dt):
            return self._dt()
        else:
            return self._dt

    @property
    def start_pos(self):
        if callable(self._start_pos):
            return self._start_pos()
        else:
            return self._start_pos

    @property
    def start_vel(self):
        if callable(self._start_vel):
            return self._start_vel()
        else:
            return self._start_vel

    def reset(self):
        self.previous_command = None


class LinearInterpolator(Repeater):
    def __init__(self, num, start_value, command_reference=False):
        super().__init__(num)
        self._start_value = start_value
        self.previous_command = None
        self.command_reference = command_reference

    def __call__(self, action):
        diff = action - self.reference
        start_pos = self.start_value

        for dt in range(self.num):
            target = start_pos + (dt+1)/self.num * diff
            yield target
        self.previous_command = action

    @property
    def reference(self):
        if self.command_reference:
            if self.previous_command is None:
                self.previous_command = self.start_value
            reference = self.previous_command
        else:
            reference = self.start_value
        return reference

    @property
    def start_value(self):
        if callable(self._start_value):
            return self._start_value()
        else:
            return self._start_value

    def reset(self):
        self.previous_command = None



class SplineInterpolator(Repeater):
    def __init__(self, num, start_value, start_velocity):
        super().__init__(num)
        self._start_value = start_value
        self._start_velocity = start_velocity

    def __call__(self, action):
        diff = action - self.reference
        t_start = 1/self.num
        dts = np.linspace(t_start, 1, self.num)
        cmds = np.stack((self.sim.joint_pos, action))
        cmd_dts = np.array([0., 1])
        res = np.zeros((self.num, 7))

        for joint in range(7):
            bc_start = (1, self.sim.joint_vel[joint])
            bc_end = (2, 0.)
            joint_cmds = cmds[:, joint]
            spline = interpolate.CubicSpline(cmd_dts, joint_cmds, bc_type=(bc_start, bc_end))
            res[:, joint] = spline(dts)

        for dt in range(self.num):
            target = res[dt, :]
            yield action

    @property
    def reference(self):
        reference = self.start_value
        return reference

    @property
    def start_velocity(self):
        if callable(self._start_velocity):
            return self._start_velocity()
        else:
            return self._start_velocity

    @property
    def start_value(self):
        if callable(self._start_value):
            return self._start_value()
        else:
            return self._start_value

    def reset(self):
        pass