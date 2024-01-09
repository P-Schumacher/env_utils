from myosuite.envs.myo.myomimic.myomimic_v0 import TrackEnvWalk
import gym
from myosuite.envs import env_base
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.tcdm.reference_motion import ReferenceMotion
from myosuite.utils.quat_math import quat2euler, euler2quat, quatDiff2Vel, mat2quat
import numpy as np
import os
import collections

PHASE_STEPS = 1

class TrackEnvWalkGoal(TrackEnvWalk):
    legacy_DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        # 'com_vel',
        'vel',
        'torso_angle',
        'feet_heights',
        'height',
        'feet_rel_positions',
        'robot_error',
        'norm_phase',
        'goal',
        'muscle_length',
        'muscle_velocity',
        'muscle_force'
    ]

    DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        'endeffectors',
        'muscle_force',
        'vel',
        'phase_clock',
        'target_reference',
        'target_speed',
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        # "robot_error": 15.0,
        "robot_error": 10.0,
        "done": -10,
        # "velocity_error": +5.0,
        # "velocity_error": +2.0,
        # "vel": 5.0
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super(BaseV0, self).__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.DEFAULT_CREDIT)
        self.initialized_pos = False
        self._setup(**kwargs)

    def _setup(self,
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):
        self.target_speeds = np.zeros((1010, 2))
        self.manual_reference = False
        self.ref_steps = 0
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs)
        self.sample_references()

    def legacy_sample_references(self):
        if not hasattr(self, 'old_ref_time'):
            self.old_ref_time = self.ref.reference['time'].copy()
        curr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        # names = [('run', os.path.join(curr_dir, 'Subj05run_99IK.pkl')), ('walk', os.path.join(curr_dir, 'Subj05walk_09IK.pkl'))]
        names = [('run', os.path.join(curr_dir, 'Subj05run_99IK.pkl')), ('jump', os.path.join(curr_dir, 'Subj05jumpIK.pkl'))]
        self.refs = {}
        for name in names:
            self.refs[name[0]] = ReferenceMotion(reference_data=name[1], motion_extrapolation=True,
                                       random_generator=self.np_random)
            self.refs[name[0]].reference['time'] = np.round(self.refs[name[0]].reference['time'], 3)
        self.target_references = []
        self.target_speeds = []
        init = 0
        for i in range(1010):
            round_time = np.round(self.old_ref_time[i * PHASE_STEPS], 3)
            if not i % 200:
                choice = np.random.choice(['run', 'jump'])
                if not init:
                    self.ref.reference['robot_init'][:] = self.refs[choice].reference['robot_init'][:]
                    init = 1
                target_vel = np.random.uniform(-1, 1, size=2)
                # target_vel = np.random.uniform(-5, 0)
            self.target_references.append(self.refs[choice].get_reference(round_time).robot)
            self.target_speeds.append(target_vel)
        self.target_references = np.array(self.target_references)
        self.target_speeds = np.array(self.target_speeds)
        self.ref.reference['robot'] = self.target_references
        self.ref.reference['time'] = np.round(self.old_ref_time[:1010], 3)

    def legacy_create_eval_reference(self, motions: list, speeds: list) -> None:
        K = 250
        assert len(motions) == len(speeds)

        curr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        files = {'run': os.path.join(curr_dir, 'Subj05run_99IK.pkl'), 'jump': os.path.join(curr_dir, 'Subj05jumpIK.pkl')}
        self.refs = {}
        for motion in motions:
            self.refs[motion] = ReferenceMotion(reference_data=files[motion], motion_extrapolation=True,
                                                 random_generator=self.np_random)
            self.refs[motion].reference['time'] = np.round(self.refs[motion].reference['time'], 3)
        self.target_references = []
        self.target_speeds = []
        self.ref.reference['robot_init'] = self.refs[motions[0]].reference['robot_init']
        idx = -1
        T = K * len(motions)
        for i in range(T):
            round_time = np.round(self.ref.reference['time'][i], 3)
            self.target_references.append(self.refs[motions[idx]].get_reference(round_time).robot)
            self.target_speeds.append(speeds[idx])
            if not i % K:
                idx += 1
        self.ref.reference['robot'] = np.array(self.target_references)
        self.ref.reference['time'] = np.array(np.round(self.ref.reference['time'][:T], 3))
        self.target_speeds = np.array(self.target_speeds)
        self.manual_reference = True

    def get_random_reference(self):
        refs = ['run', 'walk']
        choice = np.random.randint(0, 2)
        return self.refs[refs[choice]], self.vel_command[refs[choice]]

    def create_eval_reference(self) -> None:
        new_ref = ReferenceMotion(reference_data=self.reference_path, motion_extrapolation=True,
                                                random_generator=self.np_random)
        self.ref = self.get_feet(new_ref)

    def get_feet(self, new_ref):
        feets = []
        for i in range(self.ref.reference['time'].shape[0]):
            ref_mot = self.ref.reference['robot'][i]
            self.qpos_from_robot_object(self.sim.data.qpos, ref_mot)
            self.sim.forward()
            feet = self._get_endeffector_positions().reshape(-1)
            feets.append(feet)
        new_ref.reference['endeffectors'] = np.array(feets)
        return new_ref


    def get_obs_dict(self, sim):
        Nref = 10
        obs_dict = {}
        # phase for reference data
        # phase = (self.sim.data.time + self.motion_start_time ) % (self.period * self.dt)
        # norm_phase = phase / (self.period * self.dt)
        vel = self.sim.data.body('root').cvel[3:5]

        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()
        obs_dict['qvel'] = sim.data.qvel.copy()
        # obs_dict['muscle_length'] = self.muscle_lengths()
        # obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()
        obs_dict['endeffectors'] = self._get_endeffector_positions().reshape(-1).copy()
        # obs_dict['vel'] = np.array([self.sim.data.body('root').cvel[3:5]]).copy()
        obs_dict['vel'] = np.array(vel).copy()

        # Mocap ##############################################3

        # obs_dict['phase_clock'] = np.array([(self.steps % PHASE_STEPS) / PHASE_STEPS])

        curr_ref = self.ref.reference['endeffectors'][self.steps, :]
        # obs_dict['achieved'] = self.sim.data.qpos.copy()
        obs_dict['achieved'] = obs_dict['endeffectors']
        obs_dict['desired'] = curr_ref[:].copy()
        robot_error = obs_dict['achieved'] - obs_dict['desired']
        # Zero state variables that were not tracked in mocap data
        # robot_error[~np.any(self.ref.reference['robot'], axis=0)] = 0.0
        # do not take root pos or orientation into account
        # obs_dict['robot_error'] = robot_error[7:].copy()
        obs_dict['robot_error'] = robot_error[:].copy()

        # Target reference and speed in state
        obs_dict['target_reference'] = self.ref.reference['endeffectors'][self.ref_steps:self.ref_steps+Nref, :].reshape(-1).copy()

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # vel = self.sim.data.body('root').cvel[3:5]
        # reference_error = self.similarity(np.linalg.norm(obs_dict['robot_error']), beta=1) if obs_dict['phase_clock'] == 0 else 0
        reference_error = np.linalg.sum()
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            # ('robot_error', reference_error),
            # ('velocity_error', self.similarity(np.linalg.norm(obs_dict['target_speed'][0, 0, :]-vel), beta=0.9)),
            # ('velocity_error', self.similarity(np.linalg.norm(obs_dict['target_speed'][0, 0, :]-vel), beta=1.3)),
            # Must keys
            ('sparse',  0),
            ('solved',  0),
            ('done',    self._get_done())
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def _get_endeffector_positions(self, relative=True):
        # endeffector_bodies = ['hand_r', 'hand_l', 'talus_r', 'talus_l']
        endeffector_bodies = ['pelvis', 'head', 'talus_r', 'talus_l', 'hand_r', 'hand_l']
        pelvis = self.sim.data.body('pelvis').xpos
        if relative:
            return np.array([self.get_pos(body) - pelvis for body in endeffector_bodies])
        else:
            return np.array([self.get_pos(body) for body in endeffector_bodies])

    def get_pos(self, body):
        if body == 'head':
            return self.sim.data.site('head').xpos
        return self.sim.data.body(body).xpos

    def step(self, *args, **kwargs):
        self.increment_phase_steps()
        obs, reward, done, info = super().step(*args, **kwargs)
        return obs, reward, done, info

    def reset(self):
        self.ref_steps = 0
        if not self.manual_reference:
            self.sample_references()
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs

    def increment_phase_steps(self):
        if not self.steps % PHASE_STEPS:
            self.ref_steps += 1

    def get_rot_goal_reward(self):
        current = quat2euler(self.sim.data.body('pelvis').xquat)
        current[:-1] = 0
        current = euler2quat(current)
        return self.similarity(np.linalg.norm(current - self.goal_rot_pos), beta=1)

