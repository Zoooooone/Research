import matplotlib.pyplot as plt
import numpy as np
from audio_analysis import get_audio_feature, get_questionnaire_result, fit_ques_to_audio

experiment_nums = list(range(1, 8)) + list(range(9, 22))
audio_result = get_audio_feature(experiment_nums)

ques_result = get_questionnaire_result(experiment_nums)
ques_6 = fit_ques_to_audio(audio_result, ques_result[0], experiment_nums)
ques_10 = fit_ques_to_audio(audio_result, ques_result[1], experiment_nums)
audio_result = audio_result[:, 1:]

np.save("test_data/audio_ques_csv_np/audio_feature_88.npy", audio_result)
np.savetxt("test_data/audio_ques_csv_np/audio_feature_88.csv", audio_result)

np.save("test_data/audio_ques_csv_np/ques_6.npy", ques_result[0])
np.savetxt("test_data/audio_ques_csv_np/ques_6.csv", ques_result[0])

np.save("test_data/audio_ques_csv_np/ques_10.npy", ques_result[1])
np.savetxt("test_data/audio_ques_csv_np/ques_10.csv", ques_result[1])

np.save("test_data/audio_ques_csv_np/ques_6_fit.npy", ques_6)
np.savetxt("test_data/audio_ques_csv_np/ques_6_fit.csv", ques_6)

np.save("test_data/audio_ques_csv_np/ques_10_fit.npy", ques_10)
np.savetxt("test_data/audio_ques_csv_np/ques_10_fit.csv", ques_10)
