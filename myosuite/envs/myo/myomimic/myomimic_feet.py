
import copy
from myosuite.envs import env_base
from myosuite.envs.tcdm.reference_motion import ReferenceMotion
import h5py
import mujoco
from myosuite.utils.quat_math import quat2euler, euler2quat, quatDiff2Vel, mat2quat, mulQuat
from myosuite.utils.curriculum_utils import Curriculum, TerrainCurriculum
from myosuite.utils.terrain_utils import HeightField
import numpy as np
import os
import collections
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.walk_v0 import TerrainEnvV0, WalkEnvV0
from myosuite.envs.myo.myomimic.myomimic_v0 import TrackEnv
import time
import gym
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


EPS = 0.001
class TrackEnvFeet(WalkEnvV0, TrackEnv):
# class TrackEnvFeet(TerrainEnvV0, TrackEnv):

    DEFAULT_CREDIT = """\
    Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps
        Sudeep Dasari, Abhinav Gupta, Vikash Kumar
        ICRA-2023 | https://pregrasps.github.i
    """

    DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'keypoints',
        'reference_traj',
        'muscle_length',
        'muscle_velocity',
        'muscle_force'
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 0.0,
        "feet_reward": 5.0,
        "jetpack_cost": 0.0,
        "constraint_cost": -0.1307,
        "bonus": 0.0,
        "penalty": 0,
        "done": -0,
    }
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super(BaseV0, self).__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.DEFAULT_CREDIT)

        self.initialized_pos = False
        self._setup(**kwargs)

    def _setup(self,
               reference,                       # reference target/motion for behaviors
               motion_start_time:float=0,       # useful to skip initial motion
               motion_extrapolation:bool=True,  # Hold the last frame if motion is over
               ref_traj_length:int = 5,
               fuzzy_distance = 0.2,
               references = ['run', 'jump'],
               distance_function = 'smoothkloss',
               model_type='default',
               obs_keys=DEFAULT_OBS_KEYS,
               reference_type='keypoint',
               episode_length = 300,
               reference_noise = 0,
               terrain='FLAT',
               hills_range=(0,0),
               rough_range=(0,0),
               relief_range=(0,0),
               target_randomization = False,
               rotate=False,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):
        # assume pelvis is zero site !
        self.init = False
        self.episode_length = episode_length
        self.reference_noise = reference_noise
        self.target_randomization = target_randomization
        self.rotate = rotate
        self.episode_track_error = 0.0
        self.spline_range = 0.001
        self.jetpack_curriculum = Curriculum(
                                     # rate=1/50,
                                     rate=1/100,
                                     threshold=-0.002,
                                     start=0,
                                     # end=-0.1,
                                     end=-1.0,
                                     filter_coef=0.8
                                     )
        self.spline_curriculum = Curriculum(
            rate=1/1600,
            threshold=180,
            start=0.001,
            end=0.5,
            filter_coef=0.8
        )
        self.terrain_curriculum = TerrainCurriculum(
            rate=1/50,
            threshold=150,
            start=0.001,
            end=5.0,
            filter_coef=0.8
        )
        self.heightfield = HeightField(sim=self.sim,
                                               rng=self.np_random,
                                               rough_range=rough_range,
                                               hills_range=hills_range,
                                               relief_range=relief_range) if terrain != 'FLAT' else None
        self.set_keypoints(model_type, reference_type)
        self.jump = int(self.sim.model.opt.timestep * kwargs['frame_skip'] // 0.01)
        self.episode_ref = None
        # prep reference
        self._register_references(references, motion_extrapolation, random_generator=self.np_random)
        self.active_ref = self._sample_ref_key()
        self._set_distance_function(distance_function)
        self.ref_traj_length = ref_traj_length
        self._get_strided_indices()
        self.motion_extrapolation = motion_extrapolation
        self.steps = 0
        self.target_vel = - 4.0
        ##########################################
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs)

        # Adjust horizon if not motion_extrapolation
        if motion_extrapolation == False:
            self.spec.max_episode_steps = self.references[self.active_ref].horizon # doesn't work always. WIP

        # hack because in the super()._setup the initial posture is set to the average qpos and when a step is called, it ends in a `done` state
        self.initialized_pos = True

    def _sample_ref_key(self):
        """
        Randomly chooses one of the registered references to be active.
        """
        return self.np_random.choice(list(self.references.keys()))

    def _get_current_state_for_reference(self):
        if self.reference_type == 'keypoint':
            return self._get_keypoint_positions(relative=False).reshape(-1)
        elif self.reference_type == 'robot':
            return self.qpos2ref(self.sim.data.qpos)
        else:
            raise NotImplementedError

    def _set_distance_function(self, type_str):
        distance_function_dict  = {'smoothkloss': self.smoothabs2loss,
                                   'euclidean': lambda x: 1-np.linalg.norm(x, axis=-1),
                                   'sparse': lambda x: np.sum(np.where(np.linalg.norm(x, axis=-1) < 0.1)[0].shape[0]),
                                   'logeuclid': lambda x: 1 - (np.linalg.norm(x) + np.log(np.linalg.norm(x, axis=-1)+ 0.001)),
                                   'gaussian_kernel': lambda x: 1 * np.exp(-50 * np.linalg.norm(x, axis=-1)),
                                   'gaussian_kernel_wide': lambda x: 1 * np.exp(-20 * np.linalg.norm(x, axis=-1)),
                                   'gaussian_kernel_tight': lambda x: 1 * np.exp(-10000 * np.linalg.norm(x, axis=-1)),
                                   'none': lambda x: 1,
                                  }
        self.distance_function = distance_function_dict[type_str]

    def _register_references(self, references, motion_extrapolation, random_generator):
        if references[0] == 'all':
            reference_folders = ['data/stationary/', 'data/locomotion/']
            self._register_folders_of_references([reference_folders], motion_extrapolation, random_generator)
        elif references[0] == 'locomotion':
            reference_folders = 'data/locomotion/'
            self._register_folders_of_references([reference_folders], motion_extrapolation, random_generator)
        elif references[0] == 'locomotion_slow':
            reference_folders = 'data/locomotion_slow/'
            self._register_folders_of_references([reference_folders], motion_extrapolation, random_generator)
        elif references[0] == 'stationary':
            reference_folders = 'data/stationary/'
            self._register_folders_of_references([reference_folders], motion_extrapolation, random_generator)
        else:
            self._register_list_of_references(references, motion_extrapolation, random_generator)

    def _register_folders_of_references(self, reference_folders: list, motion_extrapolation, random_generator):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.references = {}
        for folder in reference_folders:
            motion_path = os.path.join(curr_dir, folder)
            files = os.listdir(motion_path)
            for file in files:
                ref_path = os.path.join(motion_path, file)
                # lazy loading for better RAM usage
                self.references[file] = ref_path

    def _register_list_of_references(self, references, motion_extrapolation, random_generator):
        reference_naming_dict = {
            "run": "corrected_data/Subj04run_63IK.npz",
            "jump": "corrected_data/Subj07jumpIK.npz",
            "lunge": "corrected_data/Subj05lungeIK.npz",
            "squat": "corrected_data/Subj04squatIK.npz",
            "land": "corrected_data/Subj05landIK.npz",
            "walk": "corrected_data/Subj05walk_18IK.npz",
        }

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.references = {}
        for ref_key in references:
            if ref_key in reference_naming_dict:
                motion_path = reference_naming_dict[ref_key]
            else:
                motion_path = ref_key
            ref_path = curr_dir+'/data/' + motion_path
            # lazy loading for better RAM usage
            self.references[ref_key] = ref_path

    def _get_strided_indices(self):
        """
        Create relative reference indices that will be used to extract
        strided future reference data
        :return:
        """

        pos = 0
        stride = 2
        rel_indices = []
        for i in range(self.ref_traj_length):
            rel_indices.append(pos)
            pos += stride
            stride *= 2
        self.rel_ref_indices = np.array(rel_indices)

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict['time'] = np.array([self.sim.data.time])
        # because no root right now
        obs_dict['qpos'] = self.sim.data.qpos.copy()
        obs_dict['qpos_without_xy'] = self.compute_qpos_obs().copy()
        obs_dict['qvel'] = sim.data.qvel.copy()
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
        obs_dict['keypoints'] = self._get_keypoint_positions(relative=True).reshape(-1).copy()
        obs_dict['muscle_length'] = self.muscle_lengths().copy()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces().copy()

        # Mocap ##############################################3
        if not self.initialized_pos:
            self._index_sample_episode_reference()
        obs_dict['current_state'] = self._get_current_state_for_reference()
        # used for reward
        obs_dict['robot_error'] = (self.episode_ref[self.steps] - obs_dict['current_state']).copy()
        # used for observation, also continues future information
        obs_dict['reference_traj'] = (self.episode_ref[self.steps + self.rel_ref_indices] - obs_dict['current_state'][np.newaxis, :]).reshape(-1).copy()

        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()
        # self.sim.model.body_names --> body names
        return obs_dict

    def similarity(self, x, beta=1):
        return np.exp(-beta * np.square(x))

    def get_reward_dict(self, obs_dict):
        vel = self.sim.data.body('root').cvel[3]
        reference_error = obs_dict['robot_error'][0, 0, :].reshape(-1, 3)
        # vertical contribution to reference error smaller
        if self.reference_type == 'keypoint':
            reference_error[...,2] = reference_error[..., 2] * 0.5
        feet_reward = np.mean(self.distance_function(reference_error))
        jetpack_cost = self.jetpack_cost()
        self.rwd_keys_wt['jetpack_cost'] = self.jetpack_curriculum.status()
        self.spline_range = self.spline_curriculum.status()
        limit_sensors = self.get_limfrc_sensor_values()
        constraint_cost = np.mean(np.abs(limit_sensors)) if len(limit_sensors) != 0 else 0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',   1.0),
            ('bonus',  1.0),
            ('feet_reward', feet_reward),
            ('jetpack_cost', jetpack_cost),
            ('constraint_cost', constraint_cost),
            ('penalty', 1.0),
            # ('vel', self._get_vel_plateau(vel)),
            # Must keys
            ('sparse',  0),
            ('solved',  0),
            ('fallen_term', self._get_fallen_termination()),
            ('ref_term', self._get_reference_termination()),
            ('done',    self._get_done()),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self.episode_track_error += np.mean(np.square(self.obs_dict['robot_error'].reshape(-1, 3)))
        string_array = np.asarray(self.active_ref, dtype=h5py.string_dtype(encoding='utf-8'))
        info['active_ref'] = string_array
        return next_state, reward, done, info

    def get_limfrc_sensor_values(self):
        """
        This cost is based on the constraint forces of the degree of freedoms that are not
        the root joint.
        """
        return [np.abs(self.sim.data.sensor(sens_idx).data) for sens_idx in range(self.sim.model.nsensor) if
                        "limfrc" in self.sim.model.sensor(sens_idx).name]

    def get_grf_values(self):
        sim = self.sim
        l_foot = [sim.model.geom(geom_idx).name for geom_idx in range(sim.model.ngeom) if ('l_foot' in sim.model.geom(geom_idx).name or 'l_bofoot' in sim.model.geom(geom_idx).name)]
        r_foot = [sim.model.geom(geom_idx).name for geom_idx in range(sim.model.ngeom) if ('r_foot' in sim.model.geom(geom_idx).name or 'r_bofoot' in sim.model.geom(geom_idx).name)]
        feet = [l_foot, r_foot]

        forces = []
        for foot in feet:
            total_force = np.zeros((6,), dtype=np.float32)
            for i in range(sim.data.ncon):
                # Note that the contact array has more than `ncon` entries,
                contact = sim.data.contact[i]
                # contacts are measured from first geom to second geom
                cond1 = sim.model.geom(contact.geom1).name == 'floor' or sim.model.geom(contact.geom1).name == 'terrain'
                if cond1 and sim.model.geom(
                        contact.geom2).name in foot:
                    # Use internal functions to read out mj_contactForce
                    c_array = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(sim.model._model, sim.data._data, i, c_array)
                    # in contrast to visualisation, the normal force is the first element!
                    total_force += c_array
            forces.append(total_force)
        return forces

    def get_ncons_hfield(self):
        sim = self.sim
        l_foot = [sim.model.geom(geom_idx).name for geom_idx in range(sim.model.ngeom) if ('l_foot' in sim.model.geom(geom_idx).name or 'l_bofoot' in sim.model.geom(geom_idx).name)]
        r_foot = [sim.model.geom(geom_idx).name for geom_idx in range(sim.model.ngeom) if ('r_foot' in sim.model.geom(geom_idx).name or 'r_bofoot' in sim.model.geom(geom_idx).name)]
        feet = [l_foot, r_foot]

        left = 0
        right = 0
        for fidx, foot in enumerate(feet):
            total_force = np.zeros((6,), dtype=np.float32)
            for i in range(sim.data.ncon):
                # Note that the contact array has more than `ncon` entries,
                contact = sim.data.contact[i]
                # contacts are measured from first geom to second geom
                if sim.model.geom(contact.geom1).name == 'terrain' and sim.model.geom(
                        contact.geom2).name in foot:
                        if fidx == 0:
                            left += 1
                        else:
                            right += 1
        return [left, right]

    def _get_vel_plateau(self, vel):
        if vel > 0:
            return 0
        if vel < self.target_vel:
            return 1.0
        else:
            return self.similarity(vel - self.target_vel, beta=1)

    def _get_done(self):
        if self._get_reference_termination():
            return 1
        if self._get_fallen_termination():
            return 1
        return 0

    def _get_reference_termination(self):
        if self.steps < 20:
            return 0
        thresh = 0.7 if self.reference_type == 'keypoint' else 10.0
        if self.init:
            dist = np.linalg.norm(self.obs_dict['robot_error'])
            if dist > thresh:
                return True

    def _get_fallen_termination(self):
        height = self._get_height()
        if height < 0.3:
            return 1
        else:
            return 0

    def qpos_from_robot_object(self, qpos, robot):
        # qpos[:len(robot)] = self.ref2qpos(robot)
        qpos[:] = self.ref2qpos(robot)
        # qpos[:len(robot)] = val
        # qpos[len(robot):len(robot)+3] = object[:3]
        # qpos[len(robot)+3:] = quat2euler(object[3:])

    def playback(self, dt=None):
        idxs = self.references[self.active_ref].find_timeslot_in_reference(self.time)
        ref_mot = self.references[self.active_ref].get_reference(self.time)
        self.qpos_from_robot_object(self.sim.data.qpos, ref_mot.robot)
        self.obs_dict = self.get_obs_dict(self.sim)
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.sim.forward()
        if dt is None:
            self.sim.data.time = self.sim.data.time + self.dt
        else:
            self.sim.data.time = self.sim.data.time + dt
        return idxs[0] < self.references[self.active_ref].horizon-1

    def set_target_spheres(self, keypoint_pos):
        for i in range(len(self.keypoint_sites)):
            target_name = f"{self.keypoint_sites[i]}_target"
            tendon_name = f"{self.keypoint_sites[i]}_tendon"
            # self.sim.model.site(target_name).pos[:] = keypoint_pos[i*3: i*3 + 3] + self.sim.data.site('pelvis').xpos
            self.sim.model.site(target_name).pos[:] = keypoint_pos[i*3: i*3 + 3] #- keypoint_pos[:3]# + self.sim.data.site('pelvis').xpos
            # self.sim.model.site(target_name).size[0] = self.fuzzy_distance
            # self.sim.model._model.tendon(tendon_name)._rgba[-1] = 1.0

    def playback_keypoint(self, dt=None):
        start = time.time()
        self.steps = (self.steps + 1) % self.episode_length
        while time.time() - start < 0.001:
            idxs = self.references[self.active_ref].find_timeslot_in_reference(self.time)
            robot = self.references[self.active_ref].reference['robot']
            self.qpos_from_robot_object(self.sim.data.qpos, robot[self.steps])
            self.sim.forward()
        if dt is None:
            self.sim.data.time = self.sim.data.time + self.dt
        else:
            self.sim.data.time = self.sim.data.time + dt
        return idxs[0] < self.references[self.active_ref].horizon-1

    def get_randomized_initial_state(self):
        # randomize qpos coordinates
        # but dont change height or rot state
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.sim.data.qvel)
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos = self.randomize_qpos(qpos)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def randomize_qpos(self, qpos):
        return qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)

    def _advance_curricula(self):
        # self.jetpack_curriculum.update(-self.episode_track_error / (self.steps+0.01))
        if self.target_randomization:
            self.spline_curriculum.update(self.steps)
        if self.heightfield is not None:
            self.terrain_curriculum.update(self.steps)

    def reset(self, reset_qpos=None, reset_qvel=None, active_ref=None):
        self._advance_curricula()
        height_adjustment = self._maybe_sample_terrain()
        self.episode_track_error = 0.0
        self.steps = 0
        self.active_ref = self._sample_ref_key() if active_ref is None else active_ref
        rsi_qpos, rsi_qvel = self._index_sample_episode_reference()
        rsi_qpos = self.ref2qpos(rsi_qpos)
        rsi_qvel = self.ref2qvel(rsi_qvel)
        # initialization
        if reset_qpos is None and reset_qvel is None:
            if self.reset_type == 'random':
                rest_qpos, reset_qvel = self.get_randomized_initial_state()

            elif self.reset_type == 'RSI':
                reset_qpos, reset_qvel = rsi_qpos, rsi_qvel

            elif self.reset_type == 'RSI_random':
                reset_qpos, reset_qvel = rsi_qpos, rsi_qvel
                reset_qpos = self.randomize_qpos(reset_qpos)

            else:
                reset_qpos, reset_qvel = self.init_qpos, self.init_qvel

        self.robot.sync_sims(self.sim, self.sim_obsd)
        if np.abs(self.reference_noise) > 0.0001:
            self.episode_ref += self.np_random.normal(0, self.reference_noise, self.episode_ref.shape)
        reset_qpos[2] += height_adjustment
        obs = BaseV0.reset(self, reset_qpos=reset_qpos, reset_qvel=reset_qvel)
        self.init = True
        return obs


    def _randomize_reference_direction(self):
        pass


    def _get_start_step(self, loaded_ref):
        """
        Get start step below episode horizon while accounting for several steps of jumps between steps
        :return:
        """
        start_time = self.np_random.uniform(0, (loaded_ref.horizon - ((self.episode_length + np.power(2, self.ref_traj_length) + 1) * self.jump)) * 0.01)
        if start_time < 0:
            raise ValueError('Sampled start time is negative. Ensure that episde length * frameskip is not larger than \
                             your data.')
        return np.round(start_time // 0.01, 2) * 0.01

    def _sample_episode_reference(self):
        """
        create reference starting at start step until end of the episode
        time: start -> start + horizon * dt
        :return:
        """
        raise NotImplementedError
        start_step = self._get_start_step()
        ref_slice = self.references[self.active_ref].get_reference(start_step)
        rsi_qpos = ref_slice.robot
        rsi_qvel = ref_slice.robot_vel
        episode_ref = []
        # make sure 2^N steps are free after episode end for strided future
        for t in np.arange(start_step, start_step + (self.episode_length + np.power(2, self.ref_traj_length)) * self.dt, self.dt):
            episode_ref.append(self.references[self.active_ref].get_reference(t).keypoint)
        self.episode_ref = np.array(episode_ref)
        return rsi_qpos, rsi_qvel

    def _index_sample_episode_reference(self):
        """
        create reference starting at start step until end of the episode
        time: start -> start + horizon * dt
        :return:
        """
        # lazy loading for better RAM usage
        loaded_ref = ReferenceMotion(reference_data=self.references[self.active_ref], motion_extrapolation=self.motion_extrapolation,
                                     random_generator=self.np_random)
        if self.episode_ref is None:
            self.episode_ref = np.zeros(shape=(self.episode_length + np.power(2, self.ref_traj_length), self.reference_dim), dtype=np.float32)
        # get t0 of reference
        start_step = self._get_start_step(loaded_ref)
        start_idxs = loaded_ref.find_timeslot_in_reference(start_step)
        # get reference state initialization
        # extract keypoints from reference
        rsi_qpos, rsi_qvel = self._augment_reference(loaded_ref, start_idxs)
        return rsi_qpos, rsi_qvel

    def _rotate_episode_ref(self, orientation):
        self.episode_ref = self.rotate_points(self.episode_ref, orientation)

    def rotate_points(self, points, theta):
        # Reshape the points array to [N, M, 3]
        points = points.reshape(-1, points.shape[1] // 3, 3)

        # Extract x and y coordinates
        x = points[:, :, 0]
        y = points[:, :, 1]

        # Compute the rotated coordinates
        # x_rotated = x * np.cos(theta + np.pi / 2) - y * np.sin(theta + np.pi / 2)
        # y_rotated = x * np.sin(theta + np.pi / 2) + y * np.cos(theta + np.pi / 2)
        x_rotated = x * np.cos(theta) - y * np.sin(theta)
        y_rotated = x * np.sin(theta) + y * np.cos(theta)

        # Update the points array with rotated coordinates
        points[:, :, 0] = x_rotated
        points[:, :, 1] = y_rotated

        # Reshape back to the original shape
        ret = points.reshape(-1, points.shape[1] * points.shape[2])
        return ret

    def _rotate_qpos_qvel(self, qpos, qvel, theta):
        """
        Rotates qpos by angle theta around z-axis
        """
        # rotate x-y position
        x_rotated = qpos[0] * np.cos(theta) - qpos[1] * np.sin(theta)
        y_rotated = qpos[0] * np.sin(theta) + qpos[1] * np.cos(theta)
        qpos[0] = x_rotated
        qpos[1] = y_rotated

        x_rotated = qvel[0] * np.cos(theta) - qvel[1] * np.sin(theta)
        y_rotated = qvel[0] * np.sin(theta) + qvel[1] * np.cos(theta)
        qvel[0] = x_rotated
        qvel[1] = y_rotated

        # rotate orientation
        rotation_quat = np.array([0., 0, 0, 0])
        mujoco.mju_axisAngle2Quat(rotation_quat, [0, 0, 1], theta)
        qpos[3:7] = mulQuat(rotation_quat, qpos[3:7])
        return qpos, qvel

    def randomize_orientation(self, qpos, qvel, target_angle=None):
                orientation = self.np_random.uniform(0, 2 * np.pi) if target_angle is None else target_angle
                rotation_quat = np.array([0., 0, 0, 0])
                mujoco.mju_axisAngle2Quat(rotation_quat, [0, 0, 1], orientation)
                qpos[3:7] = mulQuat(rotation_quat, qpos[3:7])
                # rotate original velocity with unit direction vector
                qvel[:2] = np.array([np.cos(orientation), np.sin(orientation)]) * np.linalg.norm(qvel[:2])
                return orientation, qpos, qvel

    def get_file_vel(self):
        # get velocity in file
        vel_str = self.active_ref.split('IK')[0][-2:]
        vel = float(vel_str[0]) + float(vel_str[1]) * 0.1
        # transform kmh to ms
        init_vel = + vel * (1000 / 3600)
        return init_vel

    def _maybe_adjust_ref_for_velocity(self):
        if self.model_type == 'fixed_pelvis':
            return 0
        if 'walk' in self.active_ref or 'run' in self.active_ref:
            dt = 0.01
            horizon = self.episode_ref.shape[0]
            init_vel = self.get_file_vel()
            # we rotate later
            init_vel = np.array([0, init_vel, 0])
            pos_start = np.zeros_like(self.episode_ref[0, :].reshape(-1, 3))
            pos_end = pos_start + horizon * dt * init_vel
            self.episode_ref[:, :] = (self.episode_ref[:, :].reshape(horizon, -1, 3) - np.linspace(pos_start, pos_end, horizon)).reshape(horizon, -1)
            return - init_vel
        return 0

    def _augment_reference(self, loaded_ref, start_idxs):
        """
        augment the reference with the keypoint state
        :return:
        """
        # lazy loading for better RAM usage
        self.orientation = np.random.uniform(0, 2 * np.pi)
        for i in range(self.episode_length + np.power(2, self.ref_traj_length)):
            qpos = loaded_ref.reference['robot'][start_idxs[0] + i * self.jump]
            self.qpos_from_robot_object(self.sim.data.qpos, qpos)
            self.sim.forward()
            if self.reference_type == 'keypoint':
                self.episode_ref[i, :] = self._get_keypoint_positions(relative=False).reshape(-1)
            else:
                self.episode_ref[i, :] = qpos

        rsi_qpos = loaded_ref.reference['robot'][start_idxs[0]]
        rsi_qvel = loaded_ref.reference['robot_vel'][start_idxs[0]]
        if self.target_randomization:
            rsi_qvel[:3] = self.generate_spline_trajectory()
        else:
            rsi_qvel[:3] = self._maybe_adjust_ref_for_velocity()

        if self.rotate:
            rsi_qpos, rsi_qvel = self._rotate_qpos_qvel(rsi_qpos, rsi_qvel, self.orientation)
            # rsi_qvel[:2] = np.array([np.cos(self.orientation), np.sin(self.orientation)]) * np.linalg.norm(rsi_qvel[:2])
            self._rotate_episode_ref(self.orientation)

        return rsi_qpos, rsi_qvel

    def _get_keypoint_positions(self, relative=True):
        pelvis = self.sim.data.site('pelvis').xpos
        if relative:
            return np.array([self.sim.data.site(site).xpos - pelvis for site in self.keypoint_sites])
        else:
            return np.array([self.sim.data.site(site).xpos for site in self.keypoint_sites])

    def get_pos(self, body):
        if body == 'head':
            return self.sim.data.site('head').xpos
        return self.sim.data.body(body).xpos

    def viewer_setup(self, *args, **kwargs):
        """
        Setup the default camera
        """
        distance = 5.0
        azimuth = 90
        elevation = -15
        lookat = None
        self.sim.renderer.set_free_camera_settings(
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            lookat=lookat
        )
        render_tendon = True
        render_actuator = True
        self.sim.renderer.set_viewer_settings(
            render_actuator=render_actuator,
            render_tendon=render_tendon
        )

    def smoothabs2loss(self, x):
        # p = +0.08
        # p = +0.01
        p = 0.1
        q = 10
        a = np.abs(x)
        d = np.power(a, q)
        e = d + np.power(p, q)
        s = np.power(e, 1 / q)
        f = s - p
        return - np.einsum('ij,kj->i', f, f) + 1

    def jetpack_cost(self):
        addresses = np.where(np.array([self.sim.model.actuator(i).gaintype[0] for i in range(self.sim.model.nu)]) == 0)[0]
        if len(addresses) == 0:
            return 0
        else:
            cost = 0
            for adr in addresses:
                cost += np.abs(self.sim.data.actuator(adr).ctrl)
            return cost[0] / len(addresses)


    def ref2qpos(self, reference):
        """
        Extract the right qpos from the reference
        """
        if self.model_type == 'fixed_pelvis':
            return reference[10:]
        elif self.model_type == 'jetpack':
            updated_rot = quat2euler(reference[3:7])
            return np.concatenate([reference[:3], updated_rot,  reference[7:]])
        elif self.model_type == 'full':
            return np.concatenate([reference[:3], reference[6:]])
        else:
            raise NotImplementedError

    def ref2qvel(self, reference):
        """
        Extract the right qpos from the reference
        """
        if self.model_type == 'fixed_pelvis':
            return np.zeros_like(self.sim.data.qvel)
        elif self.model_type == 'jetpack':
            vel = np.zeros_like(self.sim.data.qvel)
            vel[:3] = reference[:3]
            return vel
        elif self.model_type == 'full':
            vel = np.zeros_like(self.sim.data.qvel)
            vel[:3] = reference[:3]
            return vel
        else:
            raise NotImplementedError

    def qpos2ref(self, qpos):
        """
        Extract the right qpos from the reference
        """
        if self.model_type == 'fixed_pelvis':
            return np.concatenate([np.zeros((10,)), qpos[:]])
        elif self.model_type == 'jetpack':
            return np.concatenate([qpos[:3], euler2quat(qpos[3:6]), qpos[9:]])
            # return np.concatenate([qpos[:3], euler2quat(qpos[3:6]), qpos[10:]])
        else:
            raise NotImplementedError

    def compute_qpos_obs(self):
        """
        Compute obseravation from qpos dependant on model type
        """
        if self.model_type == 'fixed_pelvis':
            return self.sim.data.qpos[:]
        elif self.model_type == 'jetpack':
            return self.sim.data.qpos[2:]
        elif self.model_type == 'full':
            return self.sim.data.qpos[2:]
        else:
            raise NotImplementedError

    def generate_spline_trajectory(self):
        # self.spline_range = 1.0
        n0 = 100
        nend = self.episode_ref.shape[0]
        dt = 0.01
        # [horizon, numP, 3)
        points = self.episode_ref.reshape(self.episode_ref.shape[0], -1, 3)
        # get velocity in file
        init_vel = self.get_file_vel()
        vel_x = self.np_random.uniform(-self.spline_range * init_vel, self.spline_range * init_vel)

        init_pos = points[0, 2, :]
        # Create a cubic spline trajectory in x-y space
        spline = CubicSpline(
                            [init_pos[1], init_pos[1] + n0 * dt * init_vel, init_pos[1] + nend * dt * init_vel],
                            [init_pos[0], init_pos[0], init_pos[0] + nend * dt * vel_x],
        )

        # traj_x = np.linspace(init_pos[0], init_pos[0] + vel_x * nend * dt, nend)
        traj_y = np.linspace(init_pos[0], init_pos[0] + nend * init_vel * dt, nend)
        traj_x = spline(traj_y)
        traj_y = - traj_y
        deviation = []
        for x, y in zip(traj_x, traj_y):
            direction = np.arctan(x - traj_x[0] / (y - traj_y[0] + EPS))
            deviation.append(direction)
        deviation[:5] = [deviation[6]] * 5
        traj_points = np.array([traj_x, traj_y, np.zeros_like(traj_x)]).T

        # rotation
        for idx in range(points.shape[0]):
            subset = points[idx, :, :]
            rotation = Rotation.from_euler('zyx', [deviation[idx], 0, 0], degrees=False)
            # Apply the rotation to the point cloud
            points[idx] = rotation.apply(subset)
        # translation
        translate_traj_points = np.tile(traj_points[:,np.newaxis,:], [1, points.shape[1], 1])
        # translate_traj_points[:, :, 0], translate_traj_points[:, :, 1] = translate_traj_points[:, :, 1], translate_traj_points[:,:,0]
        points = points + translate_traj_points
        self.episode_ref = points.reshape(points.shape[0], -1)
        return np.array([0, -init_vel, 0])

    def _maybe_sample_terrain(self):
            """
            Sample a new terrain if the terrain type asks for it.
            """
            if self.heightfield is None or self.terrain_curriculum.current_level == 0:
                # move heightfield down if not used
                self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
                self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])
                return 0
            else:
                self.heightfield.sample(self.np_random, self.terrain_curriculum.current_level)
                self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 1.0
                self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, -5, 0])
                # self.sim.model.geom('terrain').condim = 3
                # self.sim.model.geom('floor').conaffinity = 1
                # self.sim.model.geom('floor').contype = 1
                # self.sim.model.geom('floor').pos = np.array([0, 0, -10])
                return self.heightfield.adjust_initial_height(self.sim)

    def set_keypoints(self, model_type, reference_type):

        if model_type == 'jetpack':
            self.keypoint_sites = ['pelvis', 'knee_r', 'ankle_r', 'knee_l',
                                      'ankle_l']
        elif model_type == 'fixed_pelvis':
            self.keypoint_sites = ['pelvis', 'knee_r', 'ankle_r', 'knee_l',
                                    'ankle_l']
        elif model_type == 'full':
            self.keypoint_sites = ['pelvis', 'knee_r', 'ankle_r', 'knee_l',
                                    'ankle_l']
        else:
            raise NotImplementedError

        self.reference_dim = 38 if reference_type == 'robot' else len(self.keypoint_sites) * 3

        self.reference_type = reference_type
        self.model_type = model_type


    @property
    def stiffness(self):
        try:
            return self.sim.model.joint('lumbar_extension').stiffness[0]
        except:
            return 0

    @stiffness.setter
    def stiffness(self, stiffness):
        try:
            for name in ['lumbar_extension', 'lumbar_rotation', 'lumbar_bending']:
                    self.sim.model.joint(name).stiffness[0] = stiffness 
        except:
            pass

    @property
    def damping(self):
        try:
            return self.sim.model.joint('lumbar_extension').damping[0]
        except:
            return 0

    @stiffness.setter
    def damping(self, damping):
        try:
            for name in ['lumbar_extension', 'lumbar_rotation', 'lumbar_bending']:
                    self.sim.model.joint(name).damping[0] = damping
        except:
            pass

    @property
    def rough_range(self):
        if self.heightfield is not None:
            return self.heightfield.rough_range
        else:
            return 0

    @rough_range.setter
    def rough_range(self, rough_range: list):
        if self.heightfield is not None:
            self.heightfield.rough_range = rough_range

    @property
    def hills_range(self):
        if self.heightfield is not None:
            return self.heightfield.hills_range
        else:
            return 0

    @hills_range.setter
    def hills_range(self, hills_range: list):
        if self.heightfield is not None:
            self.heightfield.hills_range = hills_range

    @property
    def relief_range(self):
        if self.heightfield is not None:
            return self.heightfield.relief_range
        else:
            return 0

    @hills_range.setter
    def hills_range(self, relief_range: list):
        if self.heightfield is not None:
            self.heightfield.relief_range = relief_range
