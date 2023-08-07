import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

path = 'test_data/result_of_opensmile/'
days = ["%s_" % i for i in range(1, 8)]
dialogue_nums = ["%s.csv" % i for i in [j for j in range(1, 12)] + [k for k in range(13, 17)] + [19] + [30, 31, 32]]


def cal_audio_feature_mean(experiment_nums):
    audio_features_means = np.empty(shape=(0, 88))
    numbers = np.empty(shape=(0, 1))

    for experiment_num in experiment_nums:
        audio_features_mean = np.empty(shape=(0, 88))

        for day in days:
            file_names = [str(experiment_num) + '_' + day + dialogue_num for dialogue_num in dialogue_nums]
            numbers = np.vstack((numbers, (np.ones(1) * (experiment_num + int(day[:-1]) * 0.1))))
            audio_features = np.empty(shape=(0, 88))

            for file in file_names:
                try:
                    audio_feature = np.array(pd.read_csv(path + file))[0]
                except FileNotFoundError:
                    pass
                else:
                    audio_features = np.vstack((audio_features, audio_feature))

            audio_features_mean = np.vstack((audio_features_mean, np.mean(audio_features, axis=0)))

        audio_features_means = np.vstack((audio_features_means, audio_features_mean))

    audio_features_means = np.hstack((numbers, audio_features_means))

    return audio_features_means


def get_questionnaire_result(experiment_nums):
    questionnaire_excel = pd.read_excel('test_data/221111_QOL_score_test.xlsx', sheet_name='並び替え')
    questionnaire_results = questionnaire_excel[[
        '氏名',
        '過去1日間に、家族、友人、近所の人、その他の仲間とのふだんのつきあいが、身体的あるいは心理的な理由で、どのくらい妨げられましたか。',
        '過去1日間に、友人や親せきを訪ねるなど、人とのつきあいが、身体的あるいは心理的な理由で、時間的にどのくらい妨げられましたか。']]
    answer_to_score_6 = {'ぜんぜん妨げられなかった': 1, 'わずかに妨げられた': 2, '少し妨げられた': 3, 'かなり妨げられた': 4, '非常に妨げられた': 5}
    answer_to_score_10 = {'いつも': 1, 'ほとんどいつも': 2, 'ときどき': 3, 'まれに': 4, 'ぜんぜんない': 5}
    names, score_6, score_10 = [], [], []

    for number in experiment_nums:
        start_index = (number - 1) * 8
        end_index = start_index + 7

        for name in questionnaire_results.iloc[start_index:end_index, 0]:
            names.append(name)

        for answer in questionnaire_results.iloc[start_index:end_index, 1]:
            score_6.append(answer_to_score_6[answer])

        for answer in questionnaire_results.iloc[start_index:end_index, 2]:
            score_10.append(answer_to_score_10[answer])

    names = np.array(names)
    score_6 = np.array(score_6)
    score_10 = np.array(score_10)
    table = np.vstack((names, np.vstack((score_6, score_10))))

    return score_6, score_10, table.T


def SVC_train(audio, question, func):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.2)
    model = svm.SVC(kernel=func, C=2.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), y_test, y_pred


def RandomForest_train(audio, question):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.25)
    model = RandomForestClassifier(n_estimators=10)
    model = model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    return score, y_test, y_pred


def DecisionTree_train(audio, question):
    X_train, X_test, y_train, y_test = train_test_split(audio, question, test_size=0.25)
    model = DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    return score, y_test, y_pred


def plot(y_test, y_pred, font=10, mark_size=2):
    plt.scatter(list(range(len(y_test))), y_test, marker='o', label='test', s=mark_size)
    plt.scatter(list(range(len(y_pred))), y_pred, marker='x', label='prediction', s=mark_size)
    plt.xticks(np.arange(1, len(y_test), 2), fontsize=font)
    plt.yticks(np.linspace(0, 6, 7), fontsize=font)
    plt.legend(fontsize=font)


if __name__ == "__main__":
    experiment_nums = [1, 2, 3, 4, 6, 9, 10, 11]
    audio_result = cal_audio_feature_mean(experiment_nums)[:, 1:]
    audio_result_float_2 = np.around(audio_result, decimals=2)
    np.savetxt('result.csv', audio_result_float_2, fmt='%.2f')

    ques_result = get_questionnaire_result(experiment_nums)
    ques_6 = ques_result[0]
    ques_10 = ques_result[1]
    funcs = ['linear', 'poly', 'rbf', 'sigmoid']

    # print(audio_result.shape, ques_6.shape)
    plt.figure(figsize=(16, 10))

    for i in range(len(funcs)):
        model_6 = SVC_train(audio_result, ques_6, funcs[i])
        model_10 = SVC_train(audio_result, ques_10, funcs[i])
        accuracy_6, accuracy_10 = model_6[0], model_10[0]
        y_test_6, y_test_10 = model_6[1], model_10[1]
        y_pred_6, y_pred_10 = model_6[2], model_10[2]

        print(f"{'question: 6':<20}{'function: ' + funcs[i]:<20}{'accuracy = ' + str(accuracy_6):<20}")
        print(f"{'question: 10':<20}{'function: ' + funcs[i]:<20}{'accuracy = ' + str(accuracy_10):<20}")

        plt.subplot(2, 4, i + 1)
        plot(y_test_6, y_pred_6)
        plt.title('question_6_' + funcs[i])

        plt.subplot(2, 4, i + 5)
        plot(y_test_10, y_pred_10)
        plt.title('question_10_' + funcs[i])

    plt.savefig('plot_result.png')
