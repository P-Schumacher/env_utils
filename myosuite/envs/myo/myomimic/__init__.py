from gym.envs.registration import register
import collections
import os
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(__file__))

FS = 5

# Task specification format
task_spec = collections.namedtuple('task_spec',
        ['name',        # task_name
         'robot',       # robot name
         'object',      # object name
         'motion',      # motion reference file path
         ])

# MyoDM tasks
MyoLegs_task_spec = (
    task_spec(name='MyoLegJump-v0', robot='MyoLeg', object=None, motion='Subj05jumpIK.pkl'),
    task_spec(name='MyoLegLunge-v0', robot='MyoLeg', object=None, motion='Subj05lungeIK.pkl'),
    task_spec(name='MyoLegSquat-v0', robot='MyoLeg', object=None, motion='Subj05squatIK.pkl'),
    task_spec(name='MyoLegLand-v0', robot='MyoLeg', object=None, motion='Subj05landIK.pkl'),
    task_spec(name='MyoLegRun-v0', robot='MyoLeg', object=None, motion='Subj05run_99IK.pkl'),
    task_spec(name='MyoLegWalk-v0', robot='MyoLeg', object=None, motion='Subj05walk_09IK.pkl'),
)
# Register MyoHand envs using reference motion


def register_myoleg_trackref_fixed(task_name, object_name, motion_path=None):
    # Track reference motion
    # print("'"+task_name+"'", end=", ")
    id_name = task_name[-3:]
    register(
        id=f'{task_name[:-3]}_fixed{id_name}',
        entry_point='myosuite.envs.myo.myomimic.myomimic_feet:TrackEnvFeet',
        max_episode_steps=200, #50steps*40Skip*2ms = 4s
        kwargs={
            'model_path': curr_dir + '/../../../simhive/myo_sim/leg/myolegs_fixed_pelvis.xml',
            'reference': curr_dir+'/data/'+motion_path,
            'terrain': 'normal',
            'reset_type': 'RSI_random',
            'frame_skip': FS,
            'references': ['stationary'],
            'model_type': 'fixed_pelvis',
            'rotate': False
        }
    )


def register_myoleg_trackref_jetpack(task_name, object_name, motion_path=None):
    # Track reference motion
    # print("'"+task_name+"'", end=", ")
    id_name = task_name[-3:]
    register(
        id=f'{task_name[:-3]}_jetpack{id_name}',
        entry_point='myosuite.envs.myo.myomimic.myomimic_feet:TrackEnvFeet',
        max_episode_steps=200, #50steps*40Skip*2ms = 4s
        kwargs={
            'model_path': curr_dir + '/../../../simhive/myo_sim/leg/myolegs_jetpack.xml',
            'reference': curr_dir+'/data/'+motion_path,
            'terrain': 'normal',
            'reset_type': 'RSI_random',
            'frame_skip': FS,
            'references': ['stationary'],
            'model_type': 'jetpack',
        }
    )


def register_myoleg_trackref_full(task_name, object_name, motion_path=None):
    # Track reference motion
    # print("'"+task_name+"'", end=", ")
    id_name = task_name[-3:]
    print(f'{task_name[:-3]}_full{id_name}')
    register(
        id=f'{task_name[:-3]}_full{id_name}',
        entry_point='myosuite.envs.myo.myomimic.myomimic_feet:TrackEnvFeet',
        max_episode_steps=200, #50steps*40Skip*2ms = 4s
        kwargs={
            'model_path': curr_dir + '/../../../simhive/myo_sim/leg/myolegs.xml',
            'reference': curr_dir+'/data/'+motion_path,
            'terrain': 'random',
            'reset_type': 'RSI_random',
            'frame_skip': FS,
            'references': ['locomotion'], # options are locomotion, stationary, locomotion_slow or any list of files with e.g. ['locomotion/Subj08walk_45IK.npz',...]
            'model_type': 'full',
            'distance_function': 'logeuclid',
            'reference_type': 'endeffector',
            'rough_range': (0.3, 2.0),
            'hills_range': (0.3, 2),
            'relief_range': (0.1, 0.3),
        }
    )



for task_name, robot_name, object_name, motion_path in MyoLegs_task_spec:
    register_myoleg_trackref_fixed(task_name, object_name, motion_path)
    register_myoleg_trackref_jetpack(task_name, object_name, motion_path)
    register_myoleg_trackref_full(task_name, object_name, motion_path)
