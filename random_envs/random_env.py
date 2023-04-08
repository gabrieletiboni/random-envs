import gym



class RandomEnv(gym.Env):
    """Superclass for all environments
        supporting Domain Randomization
        of dynamics parameters
    """

    def __init__(self, model_path, frame_skip):
        self.sampling = None
        self.dr_training = False
        self.preferred_lr = None
        self.reward_threshold = None
        self.dyn_ind_to_name = None

    
    # Methods to override in child envs:
    # ----------------------------
    def get_search_bounds_mean(self, index):
        """Get search space for current randomized parameter at index `index`"""
        raise NotImplementedError

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for current randomized parameter at index `index`"""
        raise NotImplementedError

    def get_task(self):
        """Get current dynamics parameters"""
        raise NotImplementedError

    def set_task(self, *task):
        """Set dynamics parameters to <task>"""
        raise NotImplementedError
    # ----------------------------

    def set_random_task(self):
        """Sample and set random parameters"""
        self.set_task(*self.sample_task())

    def set_dr_training(self, flag):
        """
            If True, new dynamics parameters
            are sampled and set at each .reset() call
        """
        self.dr_training = flag

    def get_dr_training(self):
        return self.dr_training

    def set_endless(self, flag):
        """
            If True, episodes are
            never reset due to custom pruning
            (done always False in .step())

            Note: episodes will still end according
            to max_timesteps (to be confirmed)
        """
        self.endless = flag

    def get_endless(self):
        return self.endless

    def get_reward_threshold(self):
        return self.reward_threshold

    def dyn_index_to_name(self, index):
        assert self.dyn_ind_to_name is not None
        return self.dyn_ind_to_name[index]

    def set_dr_distribution(self, dr_type, distr):
        """
            Set a DR distribution

            dr_type : str
                      {uniform, truncnorm, gaussian, fullgaussian}

            distr : list of distr parameters, or dict for full gaussian
        """
        if dr_type == 'uniform':
            self._set_udr_distribution(distr)
        elif dr_type == 'truncnorm':
            self._set_truncnorm_distribution(distr)
        elif dr_type == 'gaussian':
            self._set_gaussian_distribution(distr)
        elif dr_type == 'fullgaussian':
            self._set_fullgaussian_distribution(distr['mean'], distr['cov'])
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

    def _set_udr_distribution(self, bounds):
        self.sampling = 'uniform'
        for i in range(len(bounds)//2):
            self.min_task[i] = bounds[i*2]
            self.max_task[i] = bounds[i*2 + 1]
        return

    def _set_truncnorm_distribution(self, bounds):
        self.sampling = 'truncnorm'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def _set_gaussian_distribution(self, bounds):
        self.sampling = 'gaussian'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def _set_fullgaussian_distribution(self, mean, cov):
        self.sampling = 'fullgaussian'
        self.mean_task[:] = mean
        self.cov_task = np.copy(cov)
        return

    def set_task_search_bounds(self):
        """Sets the parameter search bounds based on how they are specified in get_search_bounds_mean"""
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

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def sample_task(self):
        """Sample random dynamics parameters"""
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
            raise ValueError('sampling value of random env needs to be set before using sample_task() or set_random_task(). Set it by uploading a DR distr.')

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
        """
            Set a DR distribution from a file.

            File format sample:
                ```
                # first line is skipped and used as metadata information
                par1 par2 par3 ...
                ```
        """
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
            self._set_udr_distribution(bounds)
        elif dr_type == 'truncnorm':
            self._set_truncnorm_distribution(bounds)
        elif dr_type == 'gaussian':
            self._set_gaussian_distribution(bounds)
        else:
            raise Exception('Filename is wrongly formatted: '+str(filename))

        return
