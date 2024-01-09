import gym, myosuite
from myosuite.utils.quat_math import euler2quat, mulQuat
import numpy as np
import mujoco
import time

def set_keypoints(env):
    endeff = env.episode_ref[
                env.steps
                    ]
    env.set_target_spheres(endeff)

env = gym.make(
        'MyoLegJump_full-v0',
        distance_function='euclidean',
        reference_type='keypoint',
        episode_length=200,
        references=['locomotion'],
        target_randomization=False,
        rotate=True
        )


key_qpos=np.fromstring('-0.00417948 -0.0110741 0.737332 0.619454 0.310661 0.314747 -0.648615 -0.742233 -0.003491 0.005238 1.41342 0.307173 0.272259 0.00336204 0 1.13076 0.00853522 0.094608 0.5236 -0.090766 -0.031416 -0.01623 0 0 0.418736 0 0 0 0.000247926 0.51303 0 0 0 -0.052365 -0.5236 -0.01923 0.0027946 -0.539256', sep=' ')


# rot_off_quat = np.array([0., 0, 0, 0])
# # mujoco.mju_axisAngle2Quat(rot_off_quat, [0, 0, 1.], .01)
# rot_off_quat = euler2quat([0.,0.,1])
#
#
# root_quat = env.sim.data.qpos[3:7]


N = 200
for ep in range(100):
    env.reset()
    set_keypoints(env)
    env.sim.forward()
    time.sleep(1.01)
    angles = np.linspace(0, 2 * np.pi, N)
    # env.sim.data.qpos[:] = key_qpos
    for i in range(N):
            # angle, qpos, qvel = env.randomize_orientation(env.sim.data.qpos, env.sim.data.qvel, 0.005)
            # env.rotate_points(env.episode_ref, angle)
            # mujoco.mju_mulQuat(root_quat, rot_off_quat, root_quat)
            # env.sim.data.qpos[3:7] = mulQuat(rot_off_quat, root_quat)
            # env.sim.data.qpos[3:7] = mulQuat(root_quat, rot_off_quat)
            env.step(env.action_space.sample()) 
            set_keypoints(env)
            env.sim.forward()
            env.mj_render()
            time.sleep(0.01)
            # break



