import sys

paths = ["/Users/zone/Desktop/Research_z-Chen/Opensmile/", "C:/Users/zihen/Desktop/Research_z-Chen/Opensmile"]
sys.path.append(paths[1])

import audio_analysis as au

ex_nums = list(range(1, 8)) + list(range(9, 22))
audio = au.get_audio_feature(ex_nums)
ques = au.get_questionnaire_result(ex_nums)

print(audio.shape)
print(ques[1].shape)
print(ques[1])

ques_adjusted = au.fit_ques_to_audio(audio, ques[1], ex_nums)
print(ques_adjusted.shape)
# print(ques_adjusted)
