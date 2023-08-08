import opensmile
import os

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

f = open("PATH.txt")
paths = []
for p in f:
    p = p.rstrip("\n")
    paths.append(p)

ospath = os.getcwd()
if "zihen" in ospath:
    pre_path = paths[1]
elif "zone" in ospath:
    pre_path = paths[0]

g = os.walk(pre_path + "sound/")

for p, d, f in g:
    path, dir_list, file_list = p, d, f

for file in file_list:
    result = smile.process_file(path + file)
    print(file)
    result.to_csv("test_data/result_of_opensmile_ver2/" + file[:-3] + "csv")
