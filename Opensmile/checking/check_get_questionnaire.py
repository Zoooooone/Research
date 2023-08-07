# import numpy as np
import sys

paths = ["/Users/zone/Desktop/Research_z-Chen/Opensmile/", "C:/Users/zihen/Desktop/Research_z-Chen/Opensmile"]
sys.path.append(paths[1])

import audio_analysis as au


result = au.get_questionnaire_result(list(range(1, 22)))
print(result[2])

# np.savetxt('Opensmile/checking/excel_adjusted.csv', result, fmt='%s')
# result_2 = au.get_questionnaire_result([1, 2, 3, 4, 6, 9])
# print(result_2[1].reshape(-1, 7))
# print(au.name_to_experiment_num)
