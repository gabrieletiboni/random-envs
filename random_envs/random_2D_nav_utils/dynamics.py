import numpy as np
from gym import spaces
from  abc import ABC, abstractmethod
from .geometry import Box

class AbstractRandom2DNavigationDynamics(ABC):
    def __init__(self):
        """A class which represents the dynamics of R2DN-type env."""
        self.observation_space = self._build_ob_space()
        self.action_space = self._build_action_space()
        self.init_v_offset = self._build_init_v_offset()
        self.timestep = 0.05
        self.box_mass = 1.0
        self.max_action = self._build_max_action()
        self.goal = self._build_goal()
        self.wind = self._build_wind()
        self.dyn_ind_to_name = self._build_dyn_ind_to_name()
        self.isd_randomness = None
        self.max_isd_area = None
        self.init_box_pos_distr = None
        self.init_box_vel_distr = None
    
    def set_max_isd_area(self, max_isd_area):
        self.max_isd_area = max_isd_area

    def set_isd_randomness(self, isd_randomness):
        self.isd_randomness = isd_randomness if isd_randomness is not None else 0.0

    def get_task_dim(self):
        return len(self.dyn_ind_to_name.keys())
    
    def get_original_task(self):
        return np.array(self.wind[:self.get_task_dim()])

    @abstractmethod
    def _build_ob_space(self):
        pass
    
    @abstractmethod
    def _build_action_space(self):
        pass
    
    def _build_init_v_offset(self):
        return 0.1
    
    def _build_goal(self):
        return  np.array([0.0, 1.1], dtype=np.float32)  # Position of goal
    
    def _build_wind(self):
        return np.array(
            [0.0, 0.0], dtype=np.float32
        )  # Default action from wind (no action)
    
    @abstractmethod
    def _build_max_action(self):
        pass

    def _build_vertical_wind(self):
        return False

    def _build_dyn_ind_to_name(self):
        return (
            {0: "horizontal_wind_force"}
        )
    
    def _build_init_box_pos_distr(self):
        original = self.max_isd_area
        hscale = self.isd_randomness
        vscale = self.isd_randomness
        scaled = np.array(original)
        # scale horizontally
        #scaled[:2] = (original[:2] - np.mean(original[:2]))*hscale + np.mean(original[:2])
        scaled[:2] = original[:2]*hscale
        # set bottom
        scaled[2] = self.init_v_offset
        # scale vertically
        scaled[3] = (original[3] - self.init_v_offset)*vscale + self.init_v_offset
        return scaled
    
    def _build_init_box_vel_distr(self):
        return np.array([-0.5,0.5,-0.5,0.5])*0

    def reset(self):
        # Reset box position and velocity
        if self.init_box_pos_distr is None:
            self.init_box_pos_distr = self._build_init_box_pos_distr()
        if self.init_box_vel_distr is None:
            self.init_box_vel_distr = self._build_init_box_vel_distr()
        
        self.box_pos = np.array(np.random.uniform(low=self.init_box_pos_distr[::2], high=self.init_box_pos_distr[1::2], size=(2,)),
                                dtype=np.float32)
        self.box_vel = np.array(np.random.uniform(low=self.init_box_vel_distr[::2], high=self.init_box_vel_distr[1::2], size=(2,)),
                                dtype=np.float32)
        
        # Reset distance from goal
        self.distance_from_goal = self.get_distance(self.box_pos, self.goal)

        return self._get_state()
    
    @abstractmethod
    def step(self, action, bounding_box: Box):
        pass

    def set_task(self, *task):
        for i in self.dyn_ind_to_name.keys():
            self.wind[i] = task[i]

    def get_task(self):
        i = 2 if 1 in self.dyn_ind_to_name.keys() else 1
        return np.array(self.wind[:i])

    @abstractmethod
    def _get_state(self):
        pass

    def _get_reward(self, x):
        d = self.get_squared_distance(x, self.goal)
        return (- d - np.log(d+1e-3) + 0.8) / 100

    def get_distance(self, position, goal):
        return np.sqrt(np.sum((position - goal) ** 2))

    def get_squared_distance(self, position, goal):
        return np.sum((position - goal) ** 2)
    
    def get_actor_state_mask(self):
        return []

    @abstractmethod
    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        pass

    @abstractmethod
    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        pass

    @abstractmethod
    def get_task_upper_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        pass

class Random2DNavigationDynamics(AbstractRandom2DNavigationDynamics):
    def __init__(self):
        """A class which represents the dynamics of R2DN-type env."""
        super().__init__()
    
    def _build_ob_space(self):
        return spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
    
    def _build_action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def _build_max_action(self):
        return 2.0
    
    def step(self, action, bounding_box: Box):
        # Update the box position based on the applied force and gravity
        input_force = action * self.max_action
        total_force = input_force + self.wind

        acceleration = total_force / self.box_mass

        self.box_vel = self.box_vel + acceleration * self.timestep

        has_hit_wall = bounding_box.does_trajectory_hit(
            self.box_pos,
            self.box_vel,
            np.array([0.0, 0.0], dtype=np.float32),
            self.timestep,
        )

        self.box_pos = self.box_pos + self.box_vel * self.timestep

        reward = self._get_reward(self.box_pos) - (.0 / 100 if has_hit_wall else 0.0)
        done = has_hit_wall
        info = {"distance_from_goal": self.get_distance(self.box_pos, self.goal)}

        return self._get_state(), reward, done, info

    def get_search_bounds_mean(self, index):
        return (-2.0, 2.0)

    def get_task_lower_bound(self, index):
        return -2.

    def get_task_upper_bound(self, index):
        return 2.

    def _get_state(self):
        return np.concatenate((self.box_pos, self.box_vel))

class Random2DNavigationOnlyposDynamics(Random2DNavigationDynamics):
    def __init__(self):
        super().__init__()
    
    def get_actor_state_mask(self):
        return [2, 3]

class Random2DNavigationWithRandomvelDynamics(Random2DNavigationDynamics):
    def __init__(self):
        super().__init__()

    def _build_init_box_vel_distr(self):
        return np.array([-0.5,0.5,-0.5,0.5])*self.isd_randomness

class Random2DNavigationControlledposDynamics(AbstractRandom2DNavigationDynamics):
    def __init__(self):
        """A class which represents the dynamics of R2DN-type env."""
        super().__init__()
    
    def _build_ob_space(self):
        return spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
    
    def _build_action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def _build_max_action(self):
        return 0.02

    def _build_dyn_ind_to_name(self):
        return {0: "horizontal_wind_displacement"}
    
    def step(self, action, bounding_box: Box):
        input_delta = action * self.max_action / np.sqrt(np.sum(np.square(action)))
        total_delta = input_delta + self.wind

        has_hit_wall = bounding_box.does_line_hit(
            self.box_pos,
            self.box_pos + total_delta,
        )

        self.box_pos = self.box_pos + total_delta

        reward = self._get_reward(self.box_pos) - (0.0 / 100 if has_hit_wall else 0.0)
        done = has_hit_wall
        info = {"distance_from_goal": self.get_distance(self.box_pos, self.goal)}

        return self._get_state(), reward, done, info

    def _get_state(self):
        return np.array(self.box_pos)
    
    def get_search_bounds_mean(self, index):
        return (-0.02, 0.02)
    
    def get_task_lower_bound(self, index):
        return -0.02
    
    def get_task_upper_bound(self, index):
        return 0.02
    
class Random2DNavigationControlledposCircularWindDynamics(Random2DNavigationControlledposDynamics):
    def calc_wind(self):
        CENTER = np.array([0., .5], dtype=np.float32)
        radius = self.box_pos - CENTER
        return np.array([-radius[1], radius[0]], dtype=np.float32)*self.wind/0.5

    def step(self, action, bounding_box: Box):
        input_delta = action * self.max_action # here we don't normalize the action
        total_delta = input_delta + self.calc_wind()

        has_hit_wall = bounding_box.does_line_hit(
            self.box_pos,
            self.box_pos + total_delta,
        )

        self.box_pos = self.box_pos + total_delta

        reward = self._get_reward(self.box_pos) - (0.0 / 100 if has_hit_wall else 0.0)
        done = has_hit_wall
        info = {"distance_from_goal": self.get_distance(self.box_pos, self.goal)}

        return self._get_state(), reward, done, info