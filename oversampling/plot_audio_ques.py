import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SVMSMOTE, ADASYN
from random import sample

audio_features = np.load("test_data/audio_ques_csv_np/audio_feature_88.npy")
ques_6 = np.load("test_data/audio_ques_csv_np/ques_6_fit.npy")
# print(audio_features.shape, ques_6.shape)   # (2631, 88) (2631,)

counter = Counter(ques_6)
counter = dict(sorted(counter.items(), key=lambda item: item[0]))
# print(counter)  # Counter({1: 2048, 2: 357, 3: 169, 4: 57})
major_class_number = np.max(list(dict(counter).values()))
total_class_number = np.sum(list(dict(counter).values()))
# print("major class ratio: ", major_class_number / total_class_number)  # major class ratio:  0.7784112504751045

plot_coordinate_1, plot_coordinate_2 = sample(range(0, 88), 2)

plt.figure(dpi=600, figsize=(12, 12))
plt.subplot(1, 2, 1)
for label, _ in counter.items():
    row_index = np.where(ques_6 == label)[0]
    plt.scatter(audio_features[row_index, plot_coordinate_1], audio_features[row_index, plot_coordinate_2], label=str(label), s=2)

plt.legend()
plt.title("before")

ratio = dict(counter)
for key in ratio.keys():
    ratio[key] = (ratio[key] + major_class_number) // 2

over = SVMSMOTE(sampling_strategy=ratio, random_state=1)
audio_features_new, ques_6_new = over.fit_resample(audio_features, ques_6)
counter_1 = Counter(ques_6_new)
counter_1 = dict(sorted(counter_1.items(), key=lambda item: item[0]))
# print(counter_1)  # Counter({1: 2048, 2: 1202, 4: 737, 3: 683})
# print(audio_features_new.shape, ques_6_new.shape)  # (4670, 88) (4670,)

plt.subplot(1, 2, 2)
for label, _ in counter_1.items():
    row_index = np.where(ques_6_new == label)[0]
    plt.scatter(audio_features_new[row_index, plot_coordinate_1], audio_features_new[row_index, plot_coordinate_2], label=str(label), s=2)

plt.legend()
plt.title("after")
plt.suptitle("coordinates: (%d, %d)" % (plot_coordinate_1, plot_coordinate_2))
plt.savefig("oversampling/audio_ques_6.png")
