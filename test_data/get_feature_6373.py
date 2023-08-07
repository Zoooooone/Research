import numpy as np
import pandas as pd
import sys

paths = ["/Users/zone/Desktop/Research_z-Chen/Opensmile/", "C:/Users/zihen/Desktop/Research_z-Chen/Opensmile"]
sys.path.append(paths[0])

from audio_analysis import fit_ques_to_audio

path = '/Users/zone/Desktop/学习/大学院/研究/data_opensmile_pre_exam/result_of_opensmile_ver2/'
experiment_nums = list(range(1, 8)) + list(range(9, 22))
days = ["%s_" % i for i in range(1, 8)]
dialogue_nums = ["%s.csv" % i for i in [j for j in range(1, 12)] + [k for k in range(13, 17)] + [19] + [30, 31, 32]]


def get_audio_feature():
    numbers = np.empty(shape=(0, 1))
    audio_features = np.empty(shape=(0, 6373))

    for experiment_num in experiment_nums:

        for day in days:
            file_names = [str(experiment_num) + '_' + day + dialogue_num for dialogue_num in dialogue_nums]

            for file in file_names:
                try:
                    audio_feature = np.array(pd.read_csv(path + file).iloc[:, 3:])
                except FileNotFoundError:
                    pass
                else:
                    audio_features = np.vstack((audio_features, audio_feature))
                    numbers = np.vstack((numbers, np.ones(1) * experiment_num * 10 + int(day[:-1])))

    audio_features = np.hstack((numbers, audio_features))

    return audio_features


audio_result = get_audio_feature()
ques_6 = fit_ques_to_audio(audio_result, np.load("test_data/audio_ques_csv_np/ques_6.npy"), experiment_nums)
ques_10 = fit_ques_to_audio(audio_result, np.load("test_data/audio_ques_csv_np/ques_10.npy"), experiment_nums)
audio_result = audio_result[:, 1:]

np.save("test_data/audio_feature_new/audio_feature_6373.npy", audio_result)
np.savetxt("test_data/audio_feature_new/audio_feature_6373.csv", audio_result)

np.save("test_data/audio_feature_new/ques_6_fit.npy", ques_6)
np.savetxt("test_data/audio_feature_new/ques_6_fit.csv", ques_6)

np.save("test_data/audio_feature_new/ques_10_fit.npy", ques_10)
np.savetxt("test_data/audio_feature_new/ques_10_fit.csv", ques_10)
