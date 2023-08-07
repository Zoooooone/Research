dict1 = {"A": 20, "B": 40, "C": 30}
dict2 = {}
for value in dict1.values():
    value *= 2

# print(dict1)  # {'A': 20, 'B': 40, 'C': 30}
for k in dict1.keys():
    dict2[k] = max(dict1.values()) / 2

# print(dict2)  # {'A': 20.0, 'B': 20.0, 'C': 20.0}
