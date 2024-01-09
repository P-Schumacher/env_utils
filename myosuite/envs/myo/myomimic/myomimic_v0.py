""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com), Sudeep Dasari (sdasari@andrew.cmu.edu), Vittorio Caggiano (caggiano@gmail.com)
Source  :: https://github.com/vikashplus/myosuite
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
from myosuite.envs import env_base
from myosuite.envs.tcdm.reference_motion import ReferenceMotion
from myosuite.utils.quat_math import quat2euler, euler2quat, quatDiff2Vel, mat2quat
import numpy as np
import os
import collections
from myosuite.envs.myo.base_v0 import BaseV0
# from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.walk_v0 import TerrainEnvV0

# ToDo
# - change target to reference


class TrackEnv():

    DEFAULT_CREDIT = """\
    Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps
        Sudeep Dasari, Abhinav Gupta, Vikash Kumar
        ICRA-2023 | https://pregrasps.github.io
    """

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'robot_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 0.0,#1.0,
        "bonus": 1.0,
        "penalty": -2,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.DEFAULT_CREDIT)

        self.initialized_pos = False
        self._setup(**kwargs)


    def _setup(self,
               reference,                       # reference target/motion for behaviors
               motion_start_time:float=0,       # useful to skip initial motion
               motion_extrapolation:bool=True,  # Hold the last frame if motion is over
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               Terminate_obj_fail=False,
               Terminate_pose_fail=False,
               **kwargs):

        # prep reference
        self.reference_path = reference
        self.ref = ReferenceMotion(reference_data=reference, motion_extrapolation=motion_extrapolation, random_generator=self.np_random)
        self.motion_start_time = motion_start_time

        ##########################################
        self.lift_bonus_thresh =  0.02
        ### PRE-GRASP
        self.obj_err_scale = 50
        self.base_err_scale = 40
        self.lift_bonus_mag = 1 #2.5

        ### DEEPMIMIC
        self.qpos_reward_weight = 0.35
        self.qpos_err_scale = 5.0

        self.qvel_reward_weight = 0.05
        self.qvel_err_scale = 0.1

        # TERMINATIONS FOR OBJ TRACK
        self.obj_com_term = 0.5
        # TERMINATIONS FOR HAND-OBJ DISTANCE
        self.base_fail_thresh = .25
        self.TermObj = Terminate_obj_fail

        # TERMINATIONS FOR MIMIC
        self.qpos_fail_thresh = .75
        self.TermPose = Terminate_pose_fail
        ##########################################

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs)

        # Adjust horizon if not motion_extrapolation
        if motion_extrapolation == False:
            self.spec.max_episode_steps = self.ref.horizon # doesn't work always. WIP

        # Adjust init as per the specified key
        robot_init, object_init = self.ref.get_init()
        if robot_init is not None:
            self.init_qpos[:self.ref.robot_dim] = robot_init
        if object_init is not None:
            self.init_qpos[self.ref.robot_dim:self.ref.robot_dim+3] = object_init[:3]
            self.init_qpos[-3:] = quat2euler(object_init[3:])

        # hack because in the super()._setup the initial posture is set to the average qpos and when a step is called, it ends in a `done` state
        self.initialized_pos = True
        # if self.sim.model.nkey>0:
            # self.init_qpos[:] = self.sim.model.key_qpos[0,:]


    def get_obs_dict(self, sim):
        obs_dict = {}

        # get reference for current time (returns a named tuple)
        curr_ref = self.ref.get_reference(sim.data.time+self.motion_start_time)

        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['robot_err'] = obs_dict['qp'][:].copy() - curr_ref.robot

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        # self.sim.model.body_names --> body names
        return obs_dict


    def get_reward_dict(self, obs_dict):

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',   1.0),
            ('bonus',  1.0),
            ('penalty', 1.0),
            # Must keys
            ('sparse',  0),
            ('solved',  0),
            ('done',    False),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # print(rwd_dict['dense'], obj_com_err,rwd_dict['done'],rwd_dict['sparse'])
        return rwd_dict

    def qpos_from_robot_object(self, qpos, robot):
        qpos[:len(robot)] = robot
        # qpos[len(robot):len(robot)+3] = object[:3]
        # qpos[len(robot)+3:] = quat2euler(object[3:])


    def playback(self):
        idxs = self.ref.find_timeslot_in_reference(self.time)
        # print(f"Time {self.time} {idxs} {self.ref.horizon}")
        ref_mot = self.ref.get_reference(self.time)
        self.qpos_from_robot_object(self.sim.data.qpos, ref_mot.robot)
        self.sim.forward()
        self.sim.data.time = self.sim.data.time + self.dt
        return idxs[0] < self.ref.horizon-1


    def reset(self):
        # print("Reset")
        self.ref.reset()
        # self.ref.reference['robot'][0] = - np.linspace(0, 
        obs = super().reset(self.init_qpos, self.init_qvel)
        # print(self.time, self.sim.data.qpos)
        return obs


class TrackEnvWalk(TerrainEnvV0, TrackEnv):

    DEFAULT_CREDIT = """\
    Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps
        Sudeep Dasari, Abhinav Gupta, Vikash Kumar
        ICRA-2023 | https://pregrasps.github.io
    """

    DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'torso_angle',
        # 'feet_heights',
        # 'height',
        # 'feet_rel_positions',
        'robot_error',
        'norm_phase',
        'muscle_length',
        # 'muscle_velocity',
        'muscle_force'
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 0.0,
        "robot_error": 10.0,
        "bonus": 0.0,
        "penalty": 0,
        "done": -100,
        "vel": 5.0
        # "vel": 5.0
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
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               Terminate_obj_fail=False,
               Terminate_pose_fail=False,
               **kwargs):

        # prep reference
        self.ref = ReferenceMotion(reference_data=reference, motion_extrapolation=motion_extrapolation, random_generator=self.np_random)
        self.motion_start_time = motion_start_time
        # self.period = 200
        self.period = 200
        self.steps = 0
        # self.target_vel = - 1.0
        # self.target_vel = - 0.5
        self.target_vel = - 0.2

        # TERMINATIONS FOR MIMIC
        self.qpos_fail_thresh = .75
        self.TermPose = Terminate_pose_fail
        ##########################################
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs)

        # Adjust horizon if not motion_extrapolation
        if motion_extrapolation == False:
            self.spec.max_episode_steps = self.ref.horizon # doesn't work always. WIP

        # Adjust init as per the specified key
        robot_init, object_init = self.ref.get_init()
        if robot_init is not None:
            self.init_qpos[:self.ref.robot_dim] = robot_init
        if object_init is not None:
            self.init_qpos[self.ref.robot_dim:self.ref.robot_dim+3] = object_init[:3]
            self.init_qpos[-3:] = quat2euler(object_init[3:])

        # hack because in the super()._setup the initial posture is set to the average qpos and when a step is called, it ends in a `done` state
        self.initialized_pos = True
        # if self.sim.model.nkey>0:
            # self.init_qpos[:] = self.sim.model.key_qpos[0,:]

    def get_obs_dict(self, sim):
        obs_dict = {}

        # phase for reference data
        phase = (self.sim.data.time + self.motion_start_time ) % (self.period * self.dt)
        norm_phase = phase / (self.period * self.dt)


        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()
        obs_dict['qvel'] = sim.data.qvel.copy()
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])
        obs_dict['feet_heights'] = self._get_feet_heights().copy()
        obs_dict['height'] = np.array([self._get_height()]).copy()
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()
        obs_dict['norm_phase'] = np.array([norm_phase]).copy()
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()

        # Mocap ##############################################3
        # get reference for current time (returns a named tuple)
        curr_ref = self.ref.get_reference(phase)
        obs_dict['achieved'] = self.sim.data.qpos.copy()
        obs_dict['desired'] = curr_ref.robot.copy()
        robot_error = obs_dict['achieved'] - obs_dict['desired']
        # Zero state variables that were not tracked in mocap data
        robot_error[~np.any(self.ref.reference['robot'], axis=0)] = 0.0
        obs_dict['robot_error'] = robot_error[3:].copy()
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        # self.sim.model.body_names --> body names
        return obs_dict

    def similarity(self, x, beta=1):
        return np.exp(-beta * np.square(x))

    def get_reward_dict(self, obs_dict):
        vel = self.sim.data.body('root').cvel[3]
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',   1.0),
            ('bonus',  1.0),
            # ('robot_error', self.similarity(np.linalg.norm(obs_dict['robot_error']), beta=1.5)),
            # ('robot_error', self.similarity(np.linalg.norm(obs_dict['robot_error']), beta=1.8)),
            ('robot_error', self.similarity(np.linalg.norm(obs_dict['robot_error']), beta=1.5)),
            ('penalty', 1.0),
            ('vel', self._get_vel_plateau(vel)),
            # Must keys
            ('sparse',  0),
            ('solved',  0),
            ('done',    self._get_done()),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def _get_vel_plateau(self, vel):
        if vel > 0:
            return 0
        if vel < self.target_vel:
            return 1.0
        else:
            return self.similarity(vel - self.target_vel, beta=5)

    def _get_done(self):
        height = self._get_height()
        if height < 0.7:
            return True
        else:
            return False

    def qpos_from_robot_object(self, qpos, robot):
        qpos[:len(robot)] = robot
        # qpos[len(robot):len(robot)+3] = object[:3]
        # qpos[len(robot)+3:] = quat2euler(object[3:])

    def playback(self, dt=None):
        idxs = self.ref.find_timeslot_in_reference(self.time)
        ref_mot = self.ref.get_reference(self.time)
        self.qpos_from_robot_object(self.sim.data.qpos, ref_mot.robot)
        self.obs_dict = self.get_obs_dict(self.sim)
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.sim.forward()
        if dt is None:
            self.sim.data.time = self.sim.data.time + self.dt
        else:
            self.sim.data.time = self.sim.data.time + dt
        print(self.sim.data.time)
        # self.sim.data.site('goal').xpos[-1] = -0.1
        return idxs[0] < self.ref.horizon-1

    def get_randomized_initial_state(self):
          # randomize qpos coordinates
          # but dont change height or rot state
          qpos = self.init_qpos.copy()
          qvel = np.zeros_like(self.sim.data.qvel)
          rot_state = qpos[3:7]
          height = qpos[2]
          qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
          qpos[3:7] = rot_state
          qpos[2] = height
          return qpos, qvel

    def reset(self, reset_qpos=None, reset_qvel=None):
        self.ref.reset()
        self.steps = 0
        if reset_qpos is None and reset_qvel is None:
            if self.reset_type == 'random':
                rest_qpos, reset_qvel = self.get_randomized_initial_state()
            else:
                reset_qpos, reset_qvel = self.init_qpos, self.init_qvel
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = TerrainEnvV0.reset(self, reset_qpos=reset_qpos, reset_qvel=reset_qvel)
        return obs

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



