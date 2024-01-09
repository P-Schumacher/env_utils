import gym, myosuite
import numpy as np
import deprl
from matplotlib import pyplot as plt
import time



# use references=['stationary'] or references=['locomotion'] for different datasets
env = gym.make(
        'MyoLegJump_fixed-v0', 
        distance_function='euclidean', 
        reference_type='endeffector', 
        episode_length=1000, 
        references=['locomotion/Subj08walk_45IK.npz'], 
        )

env.reset()

dep = deprl.dep_controller.DEP()
dep.initialize(action_space=env.action_space, observation_space=env.sim.data.actuator_length)

actions = []
for ep in range(1000):
    env.reset()
    for i in range(1000):
        action = dep.step(env.sim.data.actuator_length)[0,:]
        env.step(action)
        endeff = env.episode_ref[
                env.steps
            ]
        env.set_target_spheres(endeff)
        actions.append(action)

        env.mj_render()
        time.sleep(0.01)



actions = np.array(actions)
mat = np.corrcoef(actions.T)
plt.imshow(mat)
plt.savefig('correlation_dep.pdf')
np.save('correlation_dep.npy', mat)


