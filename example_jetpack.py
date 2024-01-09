import gym, myosuite
import numpy as np


# use references=['stationary'] or references=['locomotion'] for different datasets
env = gym.make(
        'MyoLegJump_jetpack-v0', 
        distance_function='euclidean', 
        reference_type='keypoint', 
        episode_length=200, 
        references=['locomotion'], 
        )


for ep in range(100):
    env.reset()
    for i in range(200):
        env.step(env.action_space.sample())
        endeff = env.episode_ref[
                env.steps
            ]
        env.set_target_spheres(endeff)

        env.mj_render()




