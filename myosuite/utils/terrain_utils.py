import numpy as np
from enum import Enum
from typing import Optional
import os
from myosuite.utils.quat_math import quat2euler, euler2quat, quatDiff2Vel, mat2quat, euler2mat

EPS = 0.001

class TerrainTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2


class SpecialTerrains(Enum):
    RELIEF = 0


class HeightField:
    def __init__(self,
                 sim,
                 rng,
                 hills_range,
                 rough_range,
                 relief_range,
                 real_x_length=10,
                 real_y_length=40,
                 view_distance=2):
        """
        Assume square quad.
        :sim: mujoco sim object.
        :rng: np_random
        :real_length: side length of quad in real-world [m]
        :patches_per_side: how many different patches we want, relative to one side length
                           total patch number will be patches_per_side^2
        """
        assert type(view_distance) is int
        self.sim = sim
        self._init_height_points()
        self.hfield = sim.model.hfield('terrain')
        self.real_x_length = real_x_length
        self.real_y_length = real_y_length
        self.patch_size = self.hfield.data.shape[1]
        # switch because of how hfield is defined
        self.patches_per_side_x = int(self.hfield.data.shape[0] / self.patch_size)
        self.patches_per_side_y = int(self.hfield.data.shape[1] / self.patch_size)
        self.view_distance = view_distance
        self.heightmap_window = None
        self.rng = rng
        self.hills_range = hills_range
        self.rough_range = rough_range
        self.relief_range = relief_range
        self._populate_patches()

    def flatten_agent_patch(self, qpos):
        """
        Turn terrain in the patch around the agent to flat.
        """
        # convert position to map position
        pos = self.cart2map(qpos[:2])
        # get patch that belongs to the position
        i = pos[0] // self.patch_size
        j = pos[1] // self.patch_size
        self._fill_patch(i, j, terrain_type=TerrainTypes.FLAT)

    def _compute_patch_data(self, terrain_type):
        if terrain_type.name == 'FLAT':
            return np.zeros((self.patch_size, self.patch_size))
        elif terrain_type.name == 'ROUGH':
            return self._compute_rough_terrain()
        elif terrain_type.name == 'HILLY':
            return self._compute_hilly_terrain()
        elif terrain_type.name == 'RELIEF':
            raise NotImplementedError('Relief not working')
            return self._compute_relief_terrain()

        else:
            raise NotImplementedError

    def _populate_patches(self):
        generated_terrains = np.zeros((len(TerrainTypes)))
        for i in range(self.patches_per_side_x):
            for j in range(self.patches_per_side_y):
                terrain_type = self.rng.choice(TerrainTypes)
                while terrain_type == TerrainTypes.FLAT:
                    terrain_type = self.rng.choice(TerrainTypes)
                self._fill_patch(i, j, terrain_type)
        # self._normalise_field()

    def _normalise_field(self):
        data = self.hfield.data
        data[:, :] = (data[:,:] - np.min(data)) / (np.max(data) - np.min(data) + EPS)
        self.hfield.data[:, :] = data[:, :]

    def _fill_patch(self, i, j, terrain_type=TerrainTypes.FLAT):
        """
        Fill patch at position <i> ,<j> with terrain <type>
        """
        self.hfield.data[i * self.patch_size: i*self.patch_size + self.patch_size,
                    j * self.patch_size: j * self.patch_size + self.patch_size] = self._compute_patch_data(terrain_type)

    def adjust_initial_height(self, sim):
        # sim.data.qpos[2] += self.hfield.data.max()
        return 0.0
        # min_height = 1000
        # for gidx in range(sim.model.ngeom):
        #     aabb_pos, aabb_size = self.sim.model.geom_aabb[gidx, :3], self.sim.model.geom_aabb[gidx, 3:]
        #     bounding_box_height = aabb_pos[2] + self.sim.model.geom(gidx).pos[2]
        #     bounding_box_minimum = bounding_box_height - 0.5 * aabb_size[2]
        #     min_height = min(min_height, bounding_box_minimum)
        #     if bounding_box_minimum < self.hfield.data.min():
        #         sim.data.qpos[2] += np.abs(bounding_box_minimum - self.hfield.data.max())

    def get_heightmap_obs(self):
        """
        Get heightmap observation.
        """
        if self.heightmap_window is None:
            self.heightmap_window = np.zeros((10, 10))
        self._measure_height()
        return self.heightmap_window[:].flatten().copy()

    def cart2map(self,
                 points_1: list,
                 points_2: Optional[list] = None):
        """
        Transform cartesian position [m * m] to rounded map position [nrow * ncol]
        If only points_1 is given: Expects cartesian positions in [x, y] format.
        If also points_2 is given: Expects points_1 = [x1, x2, ...] points_2 = [y1, y2, ...]
        """
        delta_map = self.real_length / self.nrow
        offset = self.hfield.data.shape[0] / 2
        # x, y needs to be switched to match hfield.
        if points_2 is None:
            return np.array(points_1[::-1] / delta_map + offset, dtype=np.int16)
        else:
            ret1 = np.array(points_1[:] / delta_map + offset, dtype=np.int16)
            ret2 = np.array(points_2[:] / delta_map + offset, dtype=np.int16)
            return ret2, ret1

    def sample(self, rng=None, level=0):
        """
        Sample an entire heightfield for the episode.
        Update geom in viewer if rendering.
        """
        if not rng is None:
            self.rng = rng
        self._populate_patches()
        self.hfield.data[:, :] = self.hfield.data[:, :] + 0.05
        if hasattr(self.sim, 'renderer') and not self.sim.renderer._window is None:
            self.sim.renderer._window.update_hfield(0)

    # Patch types  ---------------
    def _compute_rough_terrain(self, full=False):
        """
        Compute data for a random noise rough terrain.
        """
        if not full:
            ncol, nrow = self.patch_size, self.patch_size
        else:
            ncol, nrow = self.ncol, self.nrow
        rough = self.rng.uniform(low=-1.0, high=1.0, size=(nrow, ncol))
        normalized_data = (rough - np.min(rough)) / (np.max(rough) - np.min(rough) + EPS)
        scalar, offset = .08, .02
        scalar = self.rng.uniform(low=self.rough_range[0], high=self.rough_range[1])
        return normalized_data * scalar - offset

    def _compute_hilly_terrain(self, full=False):
        """
        Compute data for a terrain with smooth hills.
        """
        if not full:
            ncol, nrow = self.patch_size, self.patch_size
        else:
            ncol, nrow = self.ncol, self.nrow
        frequency = self.rng.randint(10, 50)
        # frequency = 10
        scalar = self.rng.uniform(low=self.hills_range[0], high=self.hills_range[1])
        data = np.sin(np.linspace(0, frequency * np.pi, nrow * ncol) + np.pi / 2) - 1
        normalized_data = (data - data.min()) / (data.max() - data.min() + EPS)
        normalized_data = np.flip(normalized_data.reshape(nrow, ncol) * scalar, [0, 1]).reshape(nrow, ncol)
        if self.rng.uniform() < 0.5:
            normalized_data = np.rot90(normalized_data)
        return normalized_data

    def _init_height_points(self):
        """ Compute grid points at which height measurements are sampled (in base frame)
         Saves the points in ndarray of shape (self.num_height_points, 3)
        """
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        y = np.array(measured_points_y)
        x = np.array(measured_points_x)
        grid_x, grid_y = np.meshgrid(x, y)

        self.num_height_points = grid_x.size
        points = np.zeros((self.num_height_points, 3))
        points[:, 0] = grid_x.flatten()
        points[:, 1] = grid_y.flatten()
        self.height_points = points

    def _measure_height(self):
        """
        Update heights at grid points around
        model.
        """
        rot_direction = quat2euler(self.sim.data.qpos[3:7])[2]
        rot_mat = euler2mat([0, 0, rot_direction])
        # rotate points around z-direction to match model
        points = np.einsum("ij,kj->ik", self.height_points, rot_mat)
        # increase point spacing
        points = (points * self.view_distance)
        # translate points to model frame
        self.points = points + (self.sim.data.qpos[:3])
        # get x and y points
        px = self.points[:, 0]
        py = self.points[:, 1]
        # get map_index coordinates of points
        px, py = self.cart2map(px, py)
        # avoid out-of-bounds by clipping indices to map boundaries
        # -2 because we go one further and shape is 1 longer than map index
        px = np.clip(px, 0, self.hfield.data.shape[0] - 2)
        py = np.clip(py, 0, self.hfield.data.shape[1] - 2)
        heights = self.hfield.data[px, py]
        if not hasattr(self, 'length'):
            self.length = 0
        self.length += 1
        # align with egocentric view of model
        self.heightmap_window[:] = np.flipud(np.rot90(heights.reshape(10, 10), axes=(1,0)))

    @property
    def size(self):
        return self.hfield.size

    @property
    def nrow(self):
        return int(self.hfield.nrow)

    @property
    def ncol(self):
        return int(self.hfield.ncol)
