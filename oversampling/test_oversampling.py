import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SVMSMOTE, ADASYN, BorderlineSMOTE, SMOTE
from collections import Counter
from random import sample

warnings.filterwarnings("ignore")

'''データの読み取り'''
audio_features = np.load("test_data/audio_ques_csv_np/audio_feature_88.npy")
ques_6 = np.load("test_data/audio_ques_csv_np/ques_6_fit.npy")
ques_10 = np.load("test_data/audio_ques_csv_np/ques_10_fit.npy")
# print(audio_features.shape, ques_6.shape, ques_10.shape)  # (2631, 88) (2631,) (2631,)
which_ques = ques_6
counter_0 = Counter(which_ques)
counter_0 = dict(sorted(counter_0.items(), key=lambda item: item[0]))
print("classes number before oversampling: ", counter_0)

'''オーバーサンプリング'''
over = SVMSMOTE(sampling_strategy='auto')
X, y = over.fit_resample(audio_features, which_ques)
# print(X.shape, y.shape)  # (4941, 88) (4941,)
counter_1 = Counter(y)
counter_1 = dict(sorted(counter_1.items(), key=lambda item: item[0]))
print("classes number after oversampling: ", counter_1)

'''各訓練データとテストデータ集を定義する'''
X_train_1, X_train_2, X_test, y_train_1, y_train_2, y_test = np.empty(shape=(0, 1)), np.empty(shape=(0, 1)), np.empty(shape=(0, 1)), np.empty(shape=(0, 1)), np.empty(shape=(0, 1)), np.empty(shape=(0, 1))
test_index, train_index_1, train_index_2 = [], [], []


def random_extract():
    test_index = sample(range(2629), 2629 // 10)
    train_index_1 = np.delete(list(range(2629)), test_index).tolist()
    train_index_2 = np.delete(list(range(X.shape[0])), test_index).tolist()
    global X_train_1, X_train_2, X_test, y_train_1, y_train_2, y_test
    X_train_1, X_test, y_train_1, y_test = audio_features[train_index_1], audio_features[test_index], which_ques[train_index_1], which_ques[test_index]
    X_train_2, y_train_2 = X[train_index_2], y[train_index_2]


def train(X_train, y_train, X_test, algorithm=RandomForestClassifier, perimeter=0, perimeter_1=0.1):
    if algorithm == DecisionTreeClassifier:
        model = algorithm(criterion=perimeter)
    elif algorithm == (RandomForestClassifier or AdaBoostClassifier):
        model = algorithm(n_estimators=perimeter)
    elif algorithm == SVC:
        model = algorithm(kernel=perimeter, probability=True)
    elif algorithm == KNeighborsClassifier:
        model = algorithm(n_neighbors=perimeter)
    elif algorithm == GradientBoostingClassifier:
        model = algorithm(learning_rate=perimeter_1, n_estimators=perimeter)
    elif algorithm == BernoulliNB:
        model = algorithm()

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return y_pred, y_pred_proba


def cal_accuracy(X_train, y_train, X_test, y_test, print_accuracy=True):
    class_total = {}
    classes_correct = {}
    accuracy = []

    for n in y_test:
        if n not in class_total:
            class_total[n] = 1
        else:
            class_total[n] += 1

    for key in class_total.keys():
        classes_correct[key] = 0

    model = DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_test[i]:
            classes_correct[y_test[i]] += 1

    classes_correct = dict(sorted(classes_correct.items(), key=lambda item: item[0]))
    class_total = dict(sorted(class_total.items(), key=lambda item: item[0]))
    total_accuracy = sum(classes_correct.values()) / (2629 // 10)

    if print_accuracy:
        print("\ntotal_class: ", class_total)
        print("correct_class: ", classes_correct, end="\n")
        print("Total accuracy: %.3f" % total_accuracy)

    for Class in class_total.keys():
        accuracy_for_each_class = classes_correct[Class] / class_total[Class]
        accuracy.append(accuracy_for_each_class)

    return total_accuracy, np.array(accuracy), y_pred


def test(X_train, y_train, X_test, y_test):
    for _ in range(100):
        total_accuracy = []
        each_accuracy = np.empty(shape=(0, len(dict(Counter(y_test)).keys())))

        result = cal_accuracy(X_train, y_train, X_test, y_test, print_accuracy=False)
        total_accuracy.append(result[0])
        each_accuracy = np.vstack((each_accuracy, result[1]))

    total_accuracy_mean = np.mean(total_accuracy)
    each_accuracy_mean = np.mean(each_accuracy, axis=0)

    print("Total_accuracy_mean: %.3f" % total_accuracy_mean)
    for i in range(each_accuracy_mean.shape[0]):
        print("Mean value of accuracy of answer %d: %.3f" % (i + 1, each_accuracy_mean[i]))


def cal_metrics(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    '''AUCを計算する'''
    y_one_hot = label_binarize(y_test, classes=np.unique(y_test))
    auc = roc_auc_score(y_one_hot, y_pred_proba, multi_class='ovo', average='macro')

    return np.array([accuracy, precision, recall, f1, auc])


def print_metrics():
    print("--------origin--------")
    result_1 = cal_accuracy(X_train_1, y_train_1, X_test, y_test, print_accuracy=False)
    # print("Accuracy: %.3f" % result_1[0])
    metrics = cal_metrics(result_1[2], y_test)
    print("Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, AUC = %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))

    print("\n--------oversampled--------")
    result_2 = cal_accuracy(X_train_2, y_train_2, X_test, y_test, print_accuracy=False)
    # print("Accuracy: %.3f" % result_2[0])
    metrics = cal_metrics(result_2[2], y_test)
    print("Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, AUC = %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))


def show_result():
    for i in range(1, 6):
        print("\n\n********Test %d********" % i)
        random_extract()
        print_metrics()


def mean_of_metrics(algorithm, perimeter=0, perimeter_1=0):
    metrics_before = np.empty(shape=(0, 5))
    metrics_after = np.empty(shape=(0, 5))

    for _ in range(10):
        random_extract()
        y_pred_before, y_pred_proba_before = train(X_train_1, y_train_1, X_test, algorithm, perimeter, perimeter_1)
        metric_before = cal_metrics(y_test, y_pred_before, y_pred_proba_before)
        metrics_before = np.vstack((metrics_before, metric_before))

        y_pred_after, y_pred_proba_after = train(X_train_2, y_train_2, X_test, algorithm, perimeter, perimeter_1)
        metric_after = cal_metrics(y_test, y_pred_after, y_pred_proba_after)
        metrics_after = np.vstack((metrics_after, metric_after))

    mean_metrics_before = np.mean(metrics_before, axis=0)
    mean_metrics_after = np.mean(metrics_after, axis=0)

    print("\nbefore: Accuracy = %.3f, Precision = %.3f, Recall = %.3f, F1 = %.3f, AUC = %.3f" % (mean_metrics_before[0], mean_metrics_before[1], mean_metrics_before[2], mean_metrics_before[3], mean_metrics_before[4]))
    print("after: Accuracy = %.3f, Precision = %.3f, Recall = %.3f, F1 = %.3f, AUC = %.3f" % (mean_metrics_after[0], mean_metrics_after[1], mean_metrics_after[2], mean_metrics_after[3], mean_metrics_after[4]))


mean_of_metrics(KNeighborsClassifier, 3)
