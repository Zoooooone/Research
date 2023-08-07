import opensmile
import os

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

g = os.walk("/Users/zone/Desktop/学习/大学院/研究/data_opensmile_pre_exam/sound/")

for p, d, f in g:
    path, dir_list, file_list = p, d, f

for file in file_list:
    result = smile.process_file(path + file)
    print(file)
    result.to_csv("/Users/zone/Desktop/学习/大学院/研究/data_opensmile_pre_exam/result_of_opensmile_ver2/" + file[:-3] + "csv")
