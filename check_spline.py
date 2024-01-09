import gym, myosuite
from myosuite.utils.quat_math import euler2quat, mulQuat
import numpy as np
import mujoco
import time
from matplotlib import pyplot as plt

def set_keypoints(env):
    endeff = env.episode_ref[
                env.steps
                    ]
    env.set_target_spheres(endeff)

env = gym.make(
        'MyoLegJump_full-v0',
        distance_function='euclidean',
        reference_type='keypoint',
        episode_length=500,
        references=['locomotion'],
        target_randomization=True,
        rotate=False
        )
env.seed(0)

key_qpos=np.fromstring('-0.00417948 -0.0110741 0.737332 0.619454 0.310661 0.314747 -0.648615 -0.742233 -0.003491 0.005238 1.41342 0.307173 0.272259 0.00336204 0 1.13076 0.00853522 0.094608 0.5236 -0.090766 -0.031416 -0.01623 0 0 0.418736 0 0 0 0.000247926 0.51303 0 0 0 -0.052365 -0.5236 -0.01923 0.0027946 -0.539256', sep=' ')


# rot_off_quat = np.array([0., 0, 0, 0])
# # mujoco.mju_axisAngle2Quat(rot_off_quat, [0, 0, 1.], .01)
# rot_off_quat = euler2quat([0.,0.,1])
#
#
# root_quat = env.sim.data.qpos[3:7]
PLOT = False

N = 500
for ep in range(500):
    env.reset()
    env.sim.forward()
    set_keypoints(env)
    time.sleep(1.01)
    angles = np.linspace(0, 2 * np.pi, N)
    # env.sim.data.qpos[:] = key_qpos
    if PLOT:
            fig, axs = plt.subplots(1,)
            data = env.episode_ref
            axs.plot(data[:,0], data[:,1], 'x', label='head')
            axs.plot(data[:,-6], data[:,-5], 'x', label='clavicle_l')
            axs.plot(data[:,-3], data[:,-2], 'x', label='clavicle_r')
            axs.set_xlim([-5,5])
            axs.set_ylim([-5, 5])
            
            plt.show()
    else:
        for i in range(N):
                # env.sim.data.qpos[:] = env.init_qpos
                set_keypoints(env)
                env.step(env.action_space.sample()) 
                set_keypoints(env)
                # env.sim.forward()
                env.mj_render()
                time.sleep(0.01)
                # break



