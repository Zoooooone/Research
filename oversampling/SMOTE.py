import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from random import randint

# print(imblearn.__version__)
# rad_state = randint(1, 1000)
X, y = make_classification(n_samples=5000, n_classes=5, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, weights=[0.9, 0.025, 0.025, 0.025, 0.025], random_state=9)
counter = Counter(y)
counter = dict(sorted(counter.items(), key=lambda item: item[0]))
# print("random_state = ", rad_state)
print(counter, X.shape)

# print(np.where(y == 0))  # [index of value == 0 in array y, dtype, ]

'''
for label, data in counter.items():  # <class 'numpy.int32'> <class 'numpy.int32'>
    print(type(label))
'''

dot_size = 2
plt.figure(dpi=600, figsize=(18, 12))
# Fig 1
plt.subplot(2, 3, 1)
for label, _ in counter.items():
    row_index = np.where(y == label)[0]
    plt.scatter(X[row_index, 0], X[row_index, 1], label=str(label), s=dot_size)

plt.legend()
plt.title("Original data")

# Fig 2
over = SMOTE(sampling_strategy={1: 2000, 2: 2000, 3: 2000, 4: 2000}, random_state=70)
X, y = over.fit_resample(X, y)
# counter_1 = sorted(Counter(y), key=lambda item: item[0])
counter_1 = Counter(y)
counter_1 = dict(sorted(counter_1.items(), key=lambda item: item[0]))
print(counter_1)

plt.subplot(2, 3, 2)
for label, _ in counter_1.items():
    row_index = np.where(y == label)[0]
    plt.scatter(X[row_index, 0], X[row_index, 1], label=str(label), s=dot_size)

plt.legend()
plt.title("oversampling_SMOTE")

# Fig 3
X, y = make_classification(n_samples=5000, n_classes=5, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, weights=[0.9, 0.025, 0.025, 0.025, 0.025], random_state=9)
over = SMOTE(sampling_strategy={1: 500, 2: 500, 3: 500, 4: 500}, random_state=70)
under = RandomUnderSampler(sampling_strategy={0: 2000}, random_state=70)
Steps = [("o", over), ("u", under)]
pipeline = Pipeline(steps=Steps)
X, y = pipeline.fit_resample(X, y)
counter_2 = Counter(y)
counter_2 = dict(sorted(counter_2.items(), key=lambda item: item[0]))
print(counter_2)

plt.subplot(2, 3, 3)
for label, _ in counter_2.items():
    row_index = np.where(y == label)[0]
    plt.scatter(X[row_index, 0], X[row_index, 1], label=str(label), s=dot_size)

plt.legend()
plt.title("over-undersampling_SMOTE")

# Fig 4
X, y = make_classification(n_samples=5000, n_classes=5, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, weights=[0.9, 0.025, 0.025, 0.025, 0.025], random_state=9)
over = BorderlineSMOTE(sampling_strategy={1: 2000, 2: 2000, 3: 2000, 4: 2000}, random_state=70)
X, y = over.fit_resample(X, y)
counter_3 = Counter(y)
counter_3 = dict(sorted(counter_3.items(), key=lambda item: item[0]))
print(counter_3)

plt.subplot(2, 3, 4)
for label, _ in counter_3.items():
    row_index = np.where(y == label)[0]
    plt.scatter(X[row_index, 0], X[row_index, 1], label=str(label), s=dot_size)

plt.legend()
plt.title("oversampling_BorderlineSMOTE")

# Fig 5
X, y = make_classification(n_samples=5000, n_classes=5, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, weights=[0.9, 0.025, 0.025, 0.025, 0.025], random_state=9)
over = SVMSMOTE(sampling_strategy={1: 2000, 2: 2000, 3: 2000, 4: 2000}, random_state=70)
X, y = over.fit_resample(X, y)
counter_4 = Counter(y)
counter_4 = dict(sorted(counter_4.items(), key=lambda item: item[0]))
print(counter_4)

plt.subplot(2, 3, 5)
for label, _ in counter_4.items():
    row_index = np.where(y == label)[0]
    plt.scatter(X[row_index, 0], X[row_index, 1], label=str(label), s=dot_size)

plt.legend()
plt.title("oversampling_SVMSMOTE")

# Fig 5
X, y = make_classification(n_samples=5000, n_classes=5, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, weights=[0.9, 0.025, 0.025, 0.025, 0.025], random_state=9)
over = ADASYN(sampling_strategy={1: 2000, 2: 2000, 3: 2000, 4: 2000}, random_state=70)
X, y = over.fit_resample(X, y)
counter_5 = Counter(y)
counter_5 = dict(sorted(counter_5.items(), key=lambda item: item[0]))
print(counter_5)

plt.subplot(2, 3, 6)
for label, _ in counter_5.items():
    row_index = np.where(y == label)[0]
    plt.scatter(X[row_index, 0], X[row_index, 1], label=str(label), s=dot_size)

plt.legend()
plt.title("oversampling_ADASYN")

plt.savefig("oversampling/comparison_SMOTE.png")
