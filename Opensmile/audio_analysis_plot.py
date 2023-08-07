import matplotlib.pyplot as plt
import numpy as np
from audio_analysis import get_audio_feature, get_questionnaire_result, fit_ques_to_audio  # , plot
from audio_analysis import RandomForest_train, DecisionTree_train, KNN_train, SVM_train, Adaboost_train, GradientBoosting_train


experiment_nums = list(range(1, 8)) + list(range(9, 22))
audio_result = get_audio_feature(experiment_nums)

ques_result = get_questionnaire_result(experiment_nums)
ques_6 = fit_ques_to_audio(audio_result, ques_result[0], experiment_nums)
ques_10 = fit_ques_to_audio(audio_result, ques_result[1], experiment_nums)
audio_result = audio_result[:, 1:]
algorithm_names = ['RandomForest', 'DecisionTree', 'KNN', 'SVM', 'GradientBoosting']

for method in algorithm_names:
    exec(f"{method}_scores_6, {method}_scores_10 = [], []")
    exec(f"{method}_losses_6, {method}_losses_10 = [], []")

n = 20  # 同じ分類作業を繰り返す回数
font = 20  # 説明文の字の大きさ
text_font = 12
s = 180  # プロットした点の大きさ
subplot_rows = 2  # subplotの総行数
subplot_columns = 2  # subplotの総列数
k = 19

'''
def subplot(test_data, pred_data, position):
    plt.subplot(subplot_rows, subplot_columns, position)
    plot(test_data[::len(test_data) // 25], pred_data[::len(pred_data) // 25], font, s)
    plt.xlabel("features", fontsize=font)
    plt.ylabel("answer", fontsize=font)
'''


def plot_score(question_number, text_position=[n // 8, 0.3], algorithms=algorithm_names, K=k):
    plt.title("Scores_" + question_number, fontsize=font)
    plt.ylabel("accuracy", fontsize=font)
    plt.xlabel("times", fontsize=font)

    for algorithm in algorithms:
        if algorithms == ['KNN']:
            i = k - 2
        else:
            i = algorithms.index(algorithm)

        if algorithm == 'KNN':
            method_name = str(k) + algorithm[1:]
        else:
            method_name = algorithm

        exec(f"plt.plot(np.arange(1, n + 1), {algorithm}_scores_{question_number}, label=method_name, marker='.', markersize=11)")
        exec(f"plt.text(text_position[0], text_position[1] - 0.03 * i, '%s = %.4f' % (method_name, np.mean(np.array({algorithm}_scores_{question_number}))), fontsize=text_font)")

    plt.xticks(np.arange(1, n + 2, n // 5), fontsize=font)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font)
    plt.legend(fontsize=font - 10)


def plot_loss(question_number, text_position=[n // 8, 0.9], algorithms=algorithm_names, K=k):
    plt.title("Losses_" + question_number, fontsize=font)
    plt.ylabel("loss", fontsize=font)
    plt.xlabel("times", fontsize=font)

    for algorithm in algorithms:
        i = algorithms.index(algorithm)

        if algorithm == 'KNN':
            method_name = str(k) + algorithm[1:]
        else:
            method_name = algorithm

        exec(f"plt.plot(np.arange(1, n + 1), {algorithm}_losses_{question_number}, label=method_name, marker='.', markersize=11)")
        exec(f"plt.text(text_position[0], text_position[1] - 0.03 * i, '%s = %.4f' % (method_name, np.mean(np.array({algorithm}_losses_{question_number}))), fontsize=text_font)")

    plt.xticks(np.arange(1, n + 2, n // 5), fontsize=font)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=font)
    plt.legend(fontsize=font - 10)


def create_model(question_number, measure, algorithms=algorithm_names, K=k):
    for algorithm in algorithms:
        if algorithm == 'KNN':
            exec(f"model_{algorithm}_{question_number} = {algorithm}_train(audio_result, ques_{question_number}, n=k)")
        else:
            exec(f"model_{algorithm}_{question_number} = {algorithm}_train(audio_result, ques_{question_number})")

        exec(f"{algorithm}_score_{question_number}, y_test_{algorithm}_{question_number}, y_pred_{algorithm}_{question_number} = model_{algorithm}_{question_number}")
        exec(f"{algorithm}_scores_{question_number}.append({algorithm}_score_{question_number})")

        if measure == 'loss':
            exec(f"{algorithm}_loss_{question_number} = np.mean(np.abs(y_pred_{algorithm}_{question_number} - y_test_{algorithm}_{question_number}))")
            exec(f"{algorithm}_losses_{question_number}.append({algorithm}_loss_{question_number})")


if __name__ == '__main__':
    plt.figure(figsize=(12, 12))

    for j in range(1, n + 1):
        create_model('10', 'loss')

    plot_loss('10')

    plt.savefig("Opensmile/plot_result/ques_10_losses_all.png")
