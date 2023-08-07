import numpy as np
import sys

paths = ["/Users/zone/Desktop/Research_z-Chen/Opensmile/", "C:/Users/zihen/Desktop/Research_z-Chen/Opensmile"]
sys.path.append(paths[1])

import audio_analysis

audio_features = audio_analysis.get_audio_feature(list(range(1, 22)))
# print(audio_features)

audio_result_float_2 = np.around(audio_features, decimals=2)
print(audio_result_float_2[audio_result_float_2[:, 0] == 11].shape)
# np.savetxt('Opensmile/checking/result.csv', audio_result_float_2, fmt='%.2f')
