from typing import Any, Dict, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "sparse",
        #reach: bool = False,
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.object_size = 0.02
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0]) #goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.1, width=1.7, height=0.4, x_offset=-0.3) #(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_cylinder(
            body_name = 'object',
            radius = self.object_size,
            height = 5*self.object_size,
            mass = 5.0,
            position = np.array([0.0, 0.0, 5*self.object_size / 2]),
            rgba_color = np.array([0.0, 0.0, 1.0, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_glass()

        self.sim.create_sphere(
            body_name = 's1',
            radius = 0.01,
            mass = 1.0,
            position = np.array([0.0, 0.3, 1.0+0.0]),
            rgba_color = np.array([1.0,0.0,0.0,1.0]),
        )
        
        self.sim.create_sphere(
            body_name = 's2',
            radius = 0.01,
            mass = 1.0,
            position = np.array([0.0, 0.3+0.02, 1.0+0.02]),
            rgba_color = np.array([1.0,0.0,0.0,1.0]),
        )

        self.sim.create_sphere(
            body_name = 's3',
            radius = 0.01,
            mass = 1.0,
            position = np.array([0.0, 0.3-0.02, 1.0+0.02]),
            rgba_color = np.array([1.0,0.0,0.0,1.0]),
        )
        self.sim.create_sphere(
            body_name = 's4',
            radius = 0.01,
            mass = 1.0,
            position = np.array([0.02, 0.3-0.02, 1.0+0.02]),
            rgba_color = np.array([1.0,0.0,0.0,1.0]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        #if self.reach:
        ee_position = np.array(self.get_ee_position())
        return ee_position
        #else:
        #    object_position = np.array(self.sim.get_base_position("object"))
        #    return object_position

    def reset(self) -> None:
        #switch for reach goal:
        self.goal = self._sample_goal()
        self.original_pos = self.goal.copy()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, 5*self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, 5*self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        #if self.reach:
        #object_position = self.sim.get_base_position("object")
        #print('shapes succ:',achieved_goal.shape, object_position.shape)
        d = distance(achieved_goal, desired_goal)
        d1 = distance(self.original_pos,self.goal)
        #print('is success:',d < self.distance_threshold,np.array(d < self.distance_threshold, dtype=np.float64))
        return np.array(d+d1 < self.distance_threshold, dtype=np.float64)
        #else:
        #    d = distance(achieved_goal, desired_goal)
        #    return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        #if self.reach:
        #object_position = self.sim.get_base_position("object")
        #print('shapes rew:',achieved_goal.shape, object_position.shape)

        # achieved_goal = (x,y,z) location of panda arm
        # desired_goal = (x,y,z) of object (here cylinder)
        # 

        d = distance(achieved_goal, desired_goal)
        
        # implement object moving penalty:
        d1 = distance(self.original_pos,self.goal)
        #print('addl reward:', d1)

        #print('reward:',-d.astype(np.float64),-np.array(d > self.distance_threshold, dtype=np.float64))
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64) - np.array(d1 > self.distance_threshold, dtype=np.float64)
        else:
            return -d.astype(np.float64) - d1.astype(np.float64)
        #else:
        #    d = distance(achieved_goal, desired_goal)
        #    if self.reward_type == "sparse":
        #        return -np.array(d > self.distance_threshold, dtype=np.float64)
        #    else:
        #        return -d
