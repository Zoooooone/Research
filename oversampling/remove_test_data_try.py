import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

x = np.empty(shape=(0, 8))
y = np.hstack((np.ones(7) * -1, np.ones(33) * -2))

for i in range(1, 41):
    x_1 = np.ones(8) * i
    x = np.vstack((x, x_1))

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# print(X_train, X_test, y_train, y_test, sep='\n\n')

over = SMOTE()
x, y = over.fit_resample(x, y)
# print(x, y)
# 補足したデータは元の配列の下に加えるので、インテックスは乱れない
