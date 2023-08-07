import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

result = smile.process_file('test_data/audio/103_1_2.wav')
result.to_csv("Opensmile/checking/features_name.csv")

# print(smile.feature_names)
