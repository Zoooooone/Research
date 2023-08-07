dict1 = {}
key = ["a", "b", "c"]

for i in range(len(key)):
    dict1[key[i]] = i

print(type(sum(dict1.values())))
