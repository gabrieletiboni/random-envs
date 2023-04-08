from collections import OrderedDict
import os
import pdb
from os import path


import numpy as np
from gym import error, spaces
from gym.utils import seeding
import gym
from random_envs.jinja.template_renderer import TemplateRenderer
from scipy.stats import truncnorm

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.renderer = TemplateRenderer()

        if not hasattr(self, "model_args"):
            self.model_args = {}

        self.build_model()
        self.endless = False

        self.data = self.sim.data

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

        self.sampling = None
        self.dr_training = False
        self.preferred_lr = None
        self.reward_threshold = None
        self.dyn_ind_to_name = None


    def set_model_args(self, args):
        self.model_args = args

    def build_model(self):
        xml = self.renderer.render_template(self.model_path, **self.model_args)
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None
        self._viewers = {}

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def get_search_bounds_mean(self, index):
        """Get search space for current randomized parameter at index <i>
        """
        raise NotImplementedError

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for current randomized parameter at index <i>
        """
        raise NotImplementedError

    def get_task(self):
        """Get current dynamics parameters"""
        raise NotImplementedError

    def set_task(self, *task):
        """Set dynamics parameters to <task>"""
        raise NotImplementedError

    def get_full_mjstate(self, state, template):
        """Given state, returns the full mjstate"""
        raise NotImplementedError

    def get_initial_mjstate(self, state, template):
        """Given state, returns the full mjstate considering
        that we are at the beginning of the episode
        """
        raise NotImplementedError

    # -----------------------------

    # DR methods ------------------
    def set_random_task(self):
        self.set_task(*self.sample_task())

    def set_dr_training(self, flag):
        """
            If True, new dynamics parameters
            are sampled and set during .reset()
        """
        self.dr_training = flag

    def get_dr_training(self):
        return self.dr_training

    def set_endless(self, flag):
        """
            If True, episodes are
            never reset (done always False)
        """
        self.endless = flag

    def get_endless(self):
        return self.endless

    def get_reward_threshold(self):
        return self.reward_threshold

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def dyn_index_to_name(self, index):
        assert self.dyn_ind_to_name is not None
        return self.dyn_ind_to_name[index]

    def set_dr_distribution(self, dr_type, distr):
        if dr_type == 'uniform':
            self.set_udr_distribution(distr)
        elif dr_type == 'truncnorm':
            self.set_truncnorm_distribution(distr)
        elif dr_type == 'gaussian':
            self.set_gaussian_distribution(distr)
        elif dr_type == 'fullgaussian':
            self.set_fullgaussian_distribution(distr['mean'], distr['cov'])
        else:
            raise Exception('Unknown dr_type:'+str(dr_type))

    def get_dr_distribution(self):
        if self.sampling == 'uniform':
            return self.min_task, self.max_task
        elif self.sampling == 'truncnorm':
            return self.mean_task, self.stdev_task
        elif self.sampling == 'gaussian':
            raise ValueError('Not implemented')
        else:
            return None

    def set_udr_distribution(self, bounds):
        self.sampling = 'uniform'
        for i in range(len(bounds)//2):
            self.min_task[i] = bounds[i*2]
            self.max_task[i] = bounds[i*2 + 1]
        return

    def set_truncnorm_distribution(self, bounds):
        self.sampling = 'truncnorm'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def set_gaussian_distribution(self, bounds):
        self.sampling = 'gaussian'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def set_fullgaussian_distribution(self, mean, cov):
        self.sampling = 'fullgaussian'
        self.mean_task[:] = mean
        self.cov_task = np.copy(cov)
        return

    def set_task_search_bounds(self):
        """Sets the task search bounds based on how they are specified in get_search_bounds_mean"""
        dim_task = len(self.get_task())
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            self.min_task[i], self.max_task[i] = b[0], b[1]

    def get_task_search_bounds(self):
        dim_task = len(self.get_task())
        min_task = np.empty(dim_task)
        max_task = np.empty(dim_task)
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            min_task[i], max_task[i] = b[0], b[1]
        return min_task, max_task

    def sample_task(self):
        if self.sampling == 'uniform':
            return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

        elif self.sampling == 'truncnorm':
            a,b = -2, 2
            sample = []

            for i, (mean, std) in enumerate(zip(self.mean_task, self.stdev_task)):
                lower_bound = self.get_task_lower_bound(i)

                attempts = 0
                obs = truncnorm.rvs(a, b, loc=mean, scale=std)
                while obs < lower_bound:
                    obs = truncnorm.rvs(a, b, loc=mean, scale=std)

                    attempts += 1
                    if attempts > 2:
                        obs = lower_bound

                sample.append( obs )

            return np.array(sample)

        elif self.sampling == 'gaussian':
            sample = []

            for mean, std in zip(self.mean_task, self.stdev_task):

                # Assuming all parameters > 0.1
                attempts = 0
                obs = np.random.randn()*std + mean
                while obs < 0.1:
                    obs = np.random.randn()*std + mean

                    attempts += 1
                    if attempts > 2:
                        raise Exception('Not all samples were above > 0.1 after 2 attempts')

                sample.append( obs )

            return np.array(sample)

        elif self.sampling == 'fullgaussian':
            # Assumes that mean_task and cov_task are work in a normalized space [0, 4]
            sample = np.random.multivariate_normal(self.mean_task, self.cov_task)
            sample = np.clip(sample, 0, 4)

            sample = self.denormalize_parameters(sample)
            return sample

        else:
            raise ValueError('sampling value of random env needs to be set before using sample_task() or set_random_task(). Set it by uploading a DR distr from file.')

        return

    def denormalize_parameters(self, parameters):
        """Denormalize parameters back to their original space
        
            Parameters are assumed to be normalized in
            a space of [0, 4]
        """
        assert parameters.shape[0] == self.task_dim

        min_task, max_task = self.get_task_search_bounds()
        parameter_bounds = np.empty((self.task_dim, 2), float)
        parameter_bounds[:,0] = min_task
        parameter_bounds[:,1] = max_task

        orig_parameters = (parameters * (parameter_bounds[:,1]-parameter_bounds[:,0]))/4 + parameter_bounds[:,0]

        return np.array(orig_parameters)

    def load_dr_distribution_from_file(self, filename):
        dr_type = None
        bounds = None

        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            dr_type = str(next(reader)[0])
            bounds = []

            second_row = next(reader)
            for col in second_row:
                bounds.append(float(col))

        if dr_type is None or bounds is None:
            raise Exception('Unable to read file:'+str(filename))

        if len(bounds) != self.task_dim*2:
            raise Exception('The file did not contain the right number of column values')

        if dr_type == 'uniform':
            self.set_udr_distribution(bounds)
        elif dr_type == 'truncnorm':
            self.set_truncnorm_distribution(bounds)
        elif dr_type == 'gaussian':
            self.set_gaussian_distribution(bounds)
        else:
            raise Exception('Filename is wrongly formatted: '+str(filename))

        return
    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        
        try:
            self.sim.forward()
        except mujoco_py.builder.MujocoException as e:
            # pdb.set_trace()
            print('task:',self.get_task())
            print('dr_training:',self.get_dr_training())
            print('qpos:', qpos)
            print('qvel:', qvel)
            print('new_state:', new_state)
            print('old_state:', old_state)
            print('-----------------------------------')
            pdb.set_trace()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
