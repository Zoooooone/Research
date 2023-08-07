import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

path = 'test_data/result_103/'
experiment_num = "103_"
days = ["%s_" % i for i in range(1, 8)]
dialogue_nums = ["%s.csv" % i for i in [j for j in range(1, 12)] + [k for k in range(13, 17)] + [19] + [30, 31, 32]]

audio_features_mean = []

for day in days:
    file_names = [experiment_num + day + dialogue_num for dialogue_num in dialogue_nums]
    audio_features = list(np.array(pd.read_csv(path + file_names[0])))
    for file in file_names[1:]:
        '''
        if file[-6:] == "32.csv" or file[-5:] == "1.csv":
            try:
                result = np.array(pd.read_csv(path + file).iloc[[0, 2]])
            except IndexError:
                result = np.array(pd.read_csv(path + file))
        else:
        '''
        audio_feature = np.array(pd.read_csv(path + file))[0]
        audio_features.append(audio_feature)
    audio_features_mean.append(np.mean(np.vstack(audio_features), axis=0))

audio_features_mean = np.vstack(audio_features_mean)
# print(audio_features_mean)

questionnaire_excel = pd.read_excel('test_data/221111_QOL_score_test.xlsx', sheet_name='並び替え後修正')
questionnaire_excel = questionnaire_excel.drop([4, 8])
questionnaire_results = questionnaire_excel[[
    '過去1日間に、家族、友人、近所の人、その他の仲間とのふだんのつきあいが、身体的あるいは心理的な理由で、どのくらい妨げられましたか。',
    '過去1日間に、友人や親せきを訪ねるなど、人とのつきあいが、身体的あるいは心理的な理由で、時間的にどのくらい妨げられましたか。']]

answer_to_score_6 = {'ぜんぜん妨げられなかった': 1, 'わずかに妨げられた': 2, '少し妨げられた': 3, 'かなり妨げられた': 4, '非常に妨げられた': 5}
answer_to_score_10 = {'いつも': 1, 'ほとんどいつも': 2, 'ときどき': 3, 'まれに': 4, 'ぜんぜんない': 5}
score_6, score_10 = [], []

for answer in questionnaire_results.iloc[:, 0]:
    score_6.append(answer_to_score_6[answer])
for answer in questionnaire_results.iloc[:, 1]:
    score_10.append(answer_to_score_10[answer])

# print(questionnaire_excel)
# print(questionnaire_results)

# score_6 = np.array(score_6)
score_6 = np.array([1, 1, 2, 2, 2, 2, 1])
score_10 = np.array(score_10)

# print(score_6, score_10)

X_train, X_test, y_train, y_test = train_test_split(audio_features_mean, score_6, test_size=0.1, random_state=0)
model = svm.SVC(kernel='linear', C=2.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
