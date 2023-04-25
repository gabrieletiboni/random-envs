from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen, functions
import mujoco_py
import numpy as np
import sys
from random_envs.mujoco_panda.core import controllers
from random_envs.mujoco_panda.core.template_renderer import TemplateRenderer


class Environment():
    """
    :description: The base class for all simulation environments. This class
        handles creating MuJoCo simulations, building the MJCF files, rendering,
        checking joint limits, and such. In short, it acts as the interface between
        the Gym wrapper and MuJoCo.
    """
    def __init__(self, template_file, init_pos_jitter=0.1, init_vel_jitter=0.0, **kwargs):
        """
        :description: The initializer for the Environment class.
        :param template_file: name of the high-level XML template file to render.
        :param init_pos_jitter: the joint position after environment reset is
            sampled from a uniform distribution; this argument sets the range
            for the initial values. This only affects joint values (qpos of
            other DoFs is not randomized). Setting this to zero will fix the
            initial position.
        :param init_vel_jitter: same as ``init_pos_jitter``, but for the
            initial velocity
        :param **kwargs: parameters which will be passed to the Jinja template
            for rendering
        """
        self.template_file = template_file
        self.model_args = kwargs
        self.viewer = None

        self.init_pos_jitter = init_pos_jitter
        self.init_vel_jitter = init_vel_jitter

        self.build_model(kwargs)
        joint_qvel_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

        self.max_endeff_vel = 2  # 2m/s end effector speed, per robot datasheet
        # The limits in the datasheet are in deg, MuJoCo uses rad
        self.joint_qvel_min = -joint_qvel_limits
        self.joint_qvel_max = joint_qvel_limits
        self.joint_qacc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])

        # Joint torque limits, per the datasheet
        self.joint_tau_max = np.array([87, 87, 87, 87, 12, 12, 12])

        # oops, it seems we were using limits in the wrong system... hence the shift
        self.joint_qpos_shift = np.array([0, 0, 0, 0, 0, -np.pi/2, -np.pi/4])
        self.joint_qpos_min = np.array([-166, -101, -166, -176, -166, -1, -166])
        self.joint_qpos_min = self.joint_qpos_min / 180 * np.pi + self.joint_qpos_shift
        self.joint_qpos_max = np.array([166, 101, 166, -4, 166, 215, 166])
        self.joint_qpos_max = self.joint_qpos_max / 180 * np.pi + self.joint_qpos_shift

    def build_model(self, model_args):
        """
        :description: This method builds (or rebuilds) the simulation model from
            the XML file. This is called when the environment is first created
            and when it needs to be rebuild because some dynamics parameters
            were changed (indicated by the ``_needs_rebuilding`` flag in
            ``PandaGymEnvironment``).
        :param model_args: the dictionary of XML Jinja template arguments
        """
        self.model = construct_model(self.template_file, **model_args)
        self.sim = MjSim(self.model)
        self.viewer = None
        self.offscreen_context = None

        self.arm_joint_index = [self.model.get_joint_qpos_addr(f"panda0_joint{i+1}") for i in range(7)]
        try:
            self.finger_joint_index = [self.model.get_joint_qpos_addr(f"panda0_finger_joint{i+1}") for i in range(2)]
        except ValueError:
            self.finger_joint_index = []

        try:
            self.finger_act_index = [self.model.actuator_name2id(f"panda0_finger_joint_{x}") for x in ["l", "r"]]
        except ValueError:
            self.finger_act_index = []

        try:
            self.arm_act_index = [self.model.actuator_name2id(f"panda0_joint{i+1}") for i in range(7)]
        except:
            pass

        try:
            mocap_body_id = self.model._body_name2id["mocap"]
            self.mocap_control = True
        except:
            self.mocap_control = False

        Environment.reset(self)

    def render(self):
        """
        :description: Render the environment on the screen. Use ``offscreen_render``
            to render to an offscreen buffer.
        """
        if self.viewer is None:
            # Initialize the viewer first time render is called
            self._initialize_viewer()
        self.viewer.render()

    def check_joint_acceleration_limit(self):
        """
        :description: Checks if the joint acceleration is within the Panda robot
            limits, per the datasheet.
        :return: a NumPy array of 7 True/False elements, indicating if the value
            is within limits (``True``) or over the limit (``False``)
        """
        qacc = np.abs(self.sim.data.qacc[self.arm_joint_index])
        return qacc < self.joint_qacc_max

    def check_joint_velocity_limit(self):
        """
        :description: Checks if the joint velocity is within the Panda robot
            limits, per the datasheet.
        :return: a NumPy array of 7 True/False elements, indicating if the value
            is within limits (``True``) or over the limit (``False``)
        """
        qvel = self.sim.data.qvel[self.arm_joint_index]
        return np.logical_and(qvel > self.joint_qvel_min, qvel < self.joint_qvel_max)

    def check_joint_position_limit(self):
        """
        :description: Checks if the joint position is within the Panda robot
            limits, per the datasheet.
        :return: a NumPy array of 7 True/False elements, indicating if the value
            is within limits (``True``) or over the limit (``False``)
        """
        qpos = self.sim.data.qpos[self.arm_joint_index]
        return np.logical_and(qpos > self.joint_qpos_min, qpos < self.joint_qpos_max)

    def offscreen_render(self, camera, shape, depth=False):
        """
        :description: Renders the environment using the given camera.
        :param camera: the name (str) or id (int)  of the camera to use for render
        :param shape: the size of image to be rendered
        :param depth: whether to render the depth image in addition to RGB,
            defaults to ``False``
        :return: The rendered RGB image (if ``depth`` is ``False``) or tuple
            of RGB and depth images (if ``depth`` is ``True``)
        """
        if self.offscreen_context is None:
            self.offscreen_context = MjRenderContextOffscreen(self.sim, offscreen=True)
        if isinstance(camera, str):
            camera_id = self.sim.model.camera_name2id(camera)
        elif isinstance(camera, int):
            camera_id = camera
        else:
            raise ValueError(f"Unknown camera {camera}; pass name or id")
        self.offscreen_context.render(*shape, camera_id)
        res = self.offscreen_context.read_pixels(*shape, depth=depth)

        if depth:
            rgb_img, depth_img = res
            # They're flipped by default
            depth_img = depth_img[::-1]
        else:
            rgb_img = res

        # RGB is also flipped
        rgb_img = rgb_img[::-1]

        if depth:
            return rgb_img, depth_img
        else:
            return rgb_img

    def _initialize_viewer(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.distance = self.model.stat.extent * 1.5

    def apply_joint_motor_command(self, command):
        """
        :description: apply actuator commands, depending on the control mode.
        :param command: When the robot is controlled with torque commands,
            this is an array of 7 torque values to be applied to the joints.
            When MuJoCo position actuators are used, this should be the target
            position value. In case of MoCap body control (similar to OpenAI
            Gym robotics envs), this is the target position of the MoCap body.
        """
        if self.mocap_control:
            self.reset_mocap2body_xpos()
            total_vel = np.sqrt(np.sum(command**2))
            clipped_command = command * np.clip(self.max_endeff_vel/(total_vel+1e-8), 0, 1)
            self.sim.data.mocap_pos[:] = self.sim.data.mocap_pos + clipped_command*self.sim_dt
        else:
            self.sim.data.ctrl[self.arm_act_index] = command

    @property
    def sim_dt(self):
        return self.sim.model.opt.timestep

    def apply_finger_motor_command(self, command):
        """
        :description: apply command to the finger actuators.
        :param command: the target position/velocity/force value, depending
            on the actuator type being used.
        """
        self.sim.data.ctrl[self.finger_act_index] = command

    def reset(self):
        """
        :description: Reset the environment to the initial state. Joint positions
            and velocities are distorted by random noise (sampled from a uniform
            distribution with ranges passed to the ``__init__`` function).
        """
        functions.mj_resetData(self.sim.model, self.sim.data)
        pos_noise = np.random.uniform(-self.init_pos_jitter, self.init_pos_jitter, 7)
        vel_noise = np.random.uniform(-self.init_vel_jitter, self.init_vel_jitter, 7)
        self.sim.data.qpos[self.arm_joint_index] += pos_noise
        self.sim.data.qvel[self.arm_joint_index] += vel_noise
        self.sim.forward()

    def denormalize_joint_pos(self, qpos):
        diff = (self.joint_qpos_max - self.joint_qpos_min)
        factor = (qpos+1)/2
        return factor * diff + self.joint_qpos_min

    def normalize_joint_pos(self, qpos):
        diff = (self.joint_qpos_max - self.joint_qpos_min)
        qpos = (qpos - self.joint_qpos_min) / diff
        qpos = qpos * 2 - 1
        return qpos

    @property
    def joint_pos(self):
        """
        :return: the joint position of the robot
        """
        return self.sim.data.qpos[self.arm_joint_index]

    @property
    def joint_vel(self):
        """
        :return: the joint velocity of the robot
        """
        return self.sim.data.qvel[self.arm_joint_index]

    @property
    def joint_acc(self):
        """
        :return: the joint acceleration of the robot
        """
        return self.sim.data.qacc[self.arm_joint_index]

    @property
    def gripper_quat(self):
        """
        :return: the quaternion describing the orientation on the end effector
        """
        quat = np.empty(4)
        rot_matrix = self.sim.data.get_site_xmat("panda0_end_effector")
        functions.mju_mat2Quat(quat, rot_matrix.ravel())
        return quat

    @property
    def gripper_pos_quat(self):
        """
        :return: position and orientation (quat) of the end-effector.
        """
        return np.concatenate((self.gripper_pos, self.gripper_quat))

    @property
    def gripper_pos(self):
        """
        :return: the position of the robot end-effector. When defining a new
            tool, this position is defined by the ``end_effector_shift`` value
            in the template. Visually, this position is marked with a small
            grey sphere, usually between the fingers of the robot.
        """
        return self.sim.data.get_site_xpos("panda0_end_effector")

    @property
    def gripper_vel(self):
        """
        :return: the translational velocity of the robot end-effector.
        """
        return self.sim.data.get_site_xvelp("panda0_end_effector")

    @property
    def gripper_ang_vel(self):
        """
        :return: the rotational velocity of the robot end-effector.
        """
        return self.sim.data.get_site_xvelr("panda0_end_effector")

    @property
    def finger_pos(self):
        """
        :return: the position of robot fingers, if present
        """
        return self.sim.data.qpos[self.finger_joint_index]

    @property
    def finger_vel(self):
        """
        :return: the velocity of robot fingers, if present
        """
        return self.sim.data.qvel[self.finger_joint_index]

    @property
    def time(self):
        """
        :return: simulation time since reset
        """
        return self.sim.data.time

    def get_geom_size(self, geom_name):
        # Find the proper geom ids
        geom_id = self.model._geom_name2id[geom_name]
        # [i:i+1] to return a reference to array element, not the value
        geom_size = self.model.geom_size[geom_id:geom_id+1]
        return geom_size

    def set_geom_size(self, geom_name, new_size):
        size = self.get_geom_size(geom_name)
        size[:] = new_size

    def get_contact_pair(self, geom_name1, geom_name2):
        # Find the proper geom ids
        geom_id1 = self.model._geom_name2id[geom_name1]
        geom_id2 = self.model._geom_name2id[geom_name2]

        # Find the right pair id
        pair_geom1 = self.model.pair_geom1
        pair_geom2 = self.model.pair_geom2
        pair_id = None
        for i, (g1, g2) in enumerate(zip(pair_geom1, pair_geom2)):
            if g1 == geom_id1 and g2 == geom_id2 \
               or g2 == geom_id1 and g1 == geom_id2:
                pair_id = i
                break
        if pair_id is None:
            raise KeyError("No contact between %s and %s defined."
                           % (geom_name1, geom_name2))
        return pair_id

    def get_body_mass_inertia(self, body_name):
        # Find body id
        body_id = self.model._body_name2id[body_name]
        # [i:i+1] to return a reference to array element, not the value
        mass = self.model.body_mass[body_id:body_id+1]
        inertia = self.model.body_inertia[body_id]
        return mass, inertia

    def get_pair_solref(self, geom_name1, geom_name2):
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        pair_solref = self.model.pair_solref[pair_id]
        return pair_solref

    def get_pair_friction(self, geom_name1, geom_name2):
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        pair_friction = self.model.pair_friction[pair_id]
        return pair_friction

    def set_body_mass_inertia(self, body_name, mass, inertia=None):
        cur_mass, cur_inertia = self.get_body_mass_inertia(body_name)

        # If no inertia passed, just scale it up with mass
        if inertia is None:
            inertia = cur_inertia/cur_mass*mass

        # Set mass and inertia
        cur_mass[:] = mass
        cur_inertia[:] = inertia

    def reset_mocap2body_xpos(self):
        """Resets the position and orientation of the mocap bodies to the same
        values as the bodies they're welded to.
        Taken from OpenAI Gym's Fetch simulation
        """

        if (self.sim.model.eq_type is None or
            self.sim.model.eq_obj1id is None or
            self.sim.model.eq_obj2id is None):
            return
        for eq_type, obj1_id, obj2_id in zip(self.sim.model.eq_type,
                                             self.sim.model.eq_obj1id,
                                             self.sim.model.eq_obj2id):
            if eq_type != mujoco_py.const.EQ_WELD:
                continue

            mocap_id = self.sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                # obj1 is the mocap, obj2 is the welded body
                body_idx = obj2_id
            else:
                # obj2 is the mocap, obj1 is the welded body
                mocap_id = self.sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert (mocap_id != -1)
            diff = self.sim.data.mocap_pos[mocap_id] - self.sim.data.body_xpos[body_idx]
            dist = np.sqrt(np.sum(diff**2))
            max_dist = .01

            self.sim.data.mocap_pos[mocap_id][:] = self.sim.data.body_xpos[body_idx] + \
                    diff * np.clip(max_dist/dist, 0, 1)


def construct_model(template_file, **kwargs):
    """
    :description: Build a MuJoCo simulation from given Jinja template file,
        passing ``kwargs`` as template arguments.
    :param template_file: the name of the XML template file to be rendered
    :param \*\*kwargs: keyword arguments which will be passed to the template
    :return: a MuJoCo simulation model loaded from the template
    """
    renderer = TemplateRenderer()
    xml_data = renderer.render_template(template_file, **kwargs)
    model = mujoco_py.load_model_from_xml(xml_data)
    return model

