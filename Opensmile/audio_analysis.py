import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

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

path = pre_path + 'result_of_opensmile/'
days = ["%s_" % i for i in range(1, 8)]
dialogue_nums = ["%s.csv" % i for i in [j for j in range(1, 12)] + [k for k in range(13, 17)] + [19] + [30, 31, 32]]
comparision_table = pd.read_excel(pre_path + "高齢者実験_被験者対応表_for_chen.xlsx")
name_to_experiment_num = {}

for num in comparision_table["被験者番号"]:
    name_to_experiment_num[int(num)] = comparision_table["被験者氏名"][num - 1]


def get_audio_feature(experiment_nums):
    numbers = np.empty(shape=(0, 1))
    audio_features = np.empty(shape=(0, 88))

    for experiment_num in experiment_nums:

        for day in days:
            file_names = [str(experiment_num) + '_' + day + dialogue_num for dialogue_num in dialogue_nums]

            for file in file_names:
                try:
                    audio_feature = np.array(pd.read_csv(path + file))[0]
                except FileNotFoundError:
                    pass
                else:
                    audio_features = np.vstack((audio_features, audio_feature))
                    numbers = np.vstack((numbers, np.ones(1) * experiment_num * 10 + int(day[:-1])))

    audio_features = np.hstack((numbers, audio_features))

    return audio_features


def get_questionnaire_result(experiment_nums):
    questionnaire_excel = pd.read_excel(pre_path + 'november_elderly_qol_answer_all.xlsx')
    questionnaire_results = questionnaire_excel[[
        '氏名',
        '過去1日間に、家族、友人、近所の人、その他の仲間とのふだんのつきあいが、身体的あるいは心理的な理由で、どのくらい妨げられましたか。',
        '過去1日間に、友人や親せきを訪ねるなど、人とのつきあいが、身体的あるいは心理的な理由で、時間的にどのくらい妨げられましたか。']]
    answer_to_score_6 = {'ぜんぜん妨げられなかった': 1, 'わずかに妨げられた': 2, '少し妨げられた': 3, 'かなり妨げられた': 4, '非常に妨げられた': 5}
    answer_to_score_10 = {'いつも': 1, 'ほとんどいつも': 2, 'ときどき': 3, 'まれに': 4, 'ぜんぜんない': 5}

    names, score_6, score_10 = [], [], []

    for num in experiment_nums:
        question_results_num = np.array(questionnaire_results[questionnaire_results["氏名"] == name_to_experiment_num[num]])

        for name in question_results_num[:-1, 0]:
            names.append(name)

        for answer in question_results_num[:-1, 1]:
            score_6.append(answer_to_score_6[answer])

        for answer in question_results_num[:-1, 2]:
            score_10.append(answer_to_score_10[answer])

    names = np.array(names)
    score_6 = np.array(score_6)
    score_10 = np.array(score_10)
    table = np.vstack((names, np.vstack((score_6, score_10))))

    return score_6, score_10, table.T


def fit_ques_to_audio(audio_features, question, experiment_nums):
    '''
    audio = [[11, ...],  question = [1, 1, 1, 1, ...],  experiment_nums = [1, 2, 3, ...]
             [11, ...],  -> question.reshape(-1, 7)
             ...
             [12, ...],
             ...
            ]
    '''
    total_answers = []
    question = question.reshape(-1, 7)

    for i in range(len(experiment_nums)):

        for j in range(1, 8):
            total_questions_for_one_day = audio_features[audio_features[:, 0] == (experiment_nums[i] * 10 + j)].shape[0]
            total_answers.extend([question[i, j - 1]] * total_questions_for_one_day)

    total_answers = np.array(total_answers)

    return total_answers


def SVM_train(audio, question, func='poly'):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.1)
    model = svm.SVC(kernel=func, C=2.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), y_test, y_pred


def RandomForest_train(audio, question):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.1)
    model = RandomForestClassifier(n_estimators=100)
    model = model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    return score, y_test, y_pred


def DecisionTree_train(audio, question, method='gini'):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.1)
    model = DecisionTreeClassifier(criterion=method)
    model = model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    return score, y_test, y_pred


def Adaboost_train(audio, question):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.1)
    model = AdaBoostClassifier(n_estimators=50)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), y_test, y_pred


def GradientBoosting_train(audio, question):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.1)
    model = GradientBoostingClassifier(n_estimators=50)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), y_test, y_pred


def KNN_train(audio, question, n=20):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.1)
    model = KNeighborsClassifier(n_neighbors=n)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    return score, y_test, y_pred


def plot(y_test, y_pred, font=10, mark_size=2):
    plt.scatter(list(range(len(y_test))), y_test, marker='o', label='test', s=mark_size)
    plt.scatter(list(range(len(y_pred))), y_pred, marker='x', label='prediction', s=mark_size)
    plt.xticks(np.arange(1, len(y_test), 2), fontsize=font)
    plt.yticks(np.linspace(0, 6, 7), fontsize=font)
    plt.legend(fontsize=font)


if __name__ == "__main__":
    experiment_nums = list(range(1, 8))  # + list(range(9, 22))
    audio_result = get_audio_feature(experiment_nums)
    ques_result = get_questionnaire_result(experiment_nums)
    ques_6 = fit_ques_to_audio(audio_result, ques_result[0], experiment_nums)
    ques_10 = fit_ques_to_audio(audio_result, ques_result[1], experiment_nums)
    funcs = ['linear', 'poly', 'rbf', 'sigmoid']

    plt.figure(figsize=(16, 10))

    for i in range(len(funcs)):
        model_6 = SVM_train(audio_result, ques_6, funcs[i])
        model_10 = SVM_train(audio_result, ques_10, funcs[i])
        accuracy_6, accuracy_10 = model_6[0], model_10[0]
        y_test_6, y_test_10 = model_6[1], model_10[1]
        y_pred_6, y_pred_10 = model_6[2], model_10[2]

        print(f"{'question: 6':<20}{'function: ' + funcs[i]:<20}{'accuracy = ' + str(accuracy_6):<20}")
        print(f"{'question: 10':<20}{'function: ' + funcs[i]:<20}{'accuracy = ' + str(accuracy_10):<20}")

        plt.subplot(2, 4, i + 1)
        plot(y_test_6[:10], y_pred_6[:10])
        plt.title('question_6_' + funcs[i])

        plt.subplot(2, 4, i + 5)
        plot(y_test_10[:10], y_pred_10[:10])
        plt.title('question_10_' + funcs[i])

    plt.savefig('plot_result.png')
