import os

g = os.walk("test_data/sound")
file_list = next(g)[-1]
print(file_list)
