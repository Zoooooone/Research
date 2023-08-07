import os

experiment_nums = list(range(1, 8)) + list(range(9, 22))
days = ["%s_" % i for i in range(1, 8)]
dialogue_nums = ["%s.csv" % i for i in [j for j in range(1, 12)] + [k for k in range(13, 17)] + [19] + [30, 31, 32]]
non_exist_files_88 = []
non_exist_files_6373 = []

for experiment_num in experiment_nums:

    for day in days:

        for dialogue_num in dialogue_nums:
            csv_filename = str(experiment_num) + '_' + day + dialogue_num
            wav_filename = str(experiment_num) + '_' + day + dialogue_num[:-3] + "wav"

            if not os.path.exists("/Users/zone/Desktop/学习/大学院/研究/data_opensmile_pre_exam/result_of_opensmile/" + csv_filename):
                non_exist_files_88.append(csv_filename)

            if not os.path.exists("/Users/zone/Desktop/学习/大学院/研究/data_opensmile_pre_exam/sound/" + wav_filename):
                non_exist_files_6373.append(wav_filename)

# print("non_exist_files_88: %d files" % len(non_exist_files_88))  # non_exist_files_88: 29 files
# print("non_exist_files_6373: %d files" % len(non_exist_files_6373))  # non_exist_files_6373: 31 files

for i in range(len(non_exist_files_6373)):
    if non_exist_files_6373[i][:-3] + "csv" not in non_exist_files_88:
        print("non_exist file: ", non_exist_files_6373[i][:-3] + "wav")
