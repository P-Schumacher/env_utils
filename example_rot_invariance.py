import mujoco
import mujoco.viewer as viewer
import os
import numpy as np
import time
curr_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(curr_dir, 'myosuite/simhive/myo_motion/myo_sim/body/myobody_Mabs.xml')
mj_model = mujoco.MjModel.from_xml_path(model_path)
mj_data = mujoco.MjData(mj_model)
window = viewer.launch_passive(mj_model, mj_data)




key_qpos=np.fromstring('-0.00417948 -0.0110741 0.737332 0.619454 0.310661 0.314747 -0.648615 -0.742233 -0.003491 0.005238 1.41342 0.307173 0.272259 0.00336204 0 1.13076 0.00853522 0.094608 0.5236 -0.090766 -0.031416 -0.01623 0 0 0.418736 0 0 0 0.000247926 0.51303 0 0 0 -0.052365 -0.5236 -0.01923 0.0027946 -0.539256', sep=' ')

mj_data.qpos[:] = key_qpos
rot_off_quat = np.array([1., 0, 0, 0])
mujoco.mju_axisAngle2Quat(rot_off_quat, [0, 0, 1.], .2)


root_quat = mj_data.qpos[3:7]
for i in range(1000000):
    mujoco.mju_mulQuat(root_quat, rot_off_quat, root_quat)
    mujoco.mj_forward(mj_model, mj_data)
    time.sleep(0.10)
    window.sync()
# input()