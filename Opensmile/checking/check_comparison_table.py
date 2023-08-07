import sys

paths = ["/Users/zone/Desktop/Research_z-Chen/Opensmile/", "C:/Users/zihen/Desktop/Research_z-Chen/Opensmile"]
sys.path.append(paths[1])

import audio_analysis as au

print(au.name_to_experiment_num)
print(len(au.name_to_experiment_num))
