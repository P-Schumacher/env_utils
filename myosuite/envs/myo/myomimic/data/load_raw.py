from matplotlib import pyplot as plt
from robohive.envs.tcdm.reference_motion import ReferenceMotion


reference_data = './'
# reference_data = reference_data + 'Subj05walk_09IK.pkl'
reference_data = reference_data + 'Subj05run_99IK.pkl'
# reference_data = reference_data + 'Subj05squatIK.pkl'

reference  = ReferenceMotion(reference_data)
# from pudb import set_trace; set_trace()
fig, axs = plt.subplots(35,1, figsize=(10, 20))
for i in range(35):
    axs[i].plot(reference.reference['time'][:3000], reference.reference['robot'][:3000, i])
print(reference.reference['time'][:3000])
plt.show()
