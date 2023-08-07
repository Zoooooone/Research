import numpy as np
np.set_printoptions(threshold=np.inf)

'''
x = np.empty(shape=(0, 8))

for i in range(10):
    y = np.ones(8) * i
    x = np.vstack((x, y))

x = x.reshape(1, -1)[0]

print(x)  # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, ...]
print(x.shape)  # (80, )


'''
p = np.array([1, 2, 3, 4, 5, 6, 1, 8, 9])
n = np.empty(shape=(0, 9))

for i in range(10):
    n = np.vstack((n, p))

n_1 = n.reshape(1, -1)
n_2 = n.T.reshape(1, -1)

print(n_1)
print(n_2)
# print(n_2 == 1)
# print(n)
# print((n[:, :1] == 1).shape)  # (10, 1)
