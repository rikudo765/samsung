import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt


def get_unigram():
    tmp = []
    cols = ['.', 't', 'i', 'e', '5', 'Key.shift', 'R', 'o', 'a', 'n', 'l', 'Key.enter']

    os.chdir("/home/bohdan/samsung/2020-knu-ka/classifiers/team_data")

    for file_name in os.listdir("unigrams"):
        uni = pd.read_csv("unigrams/" + file_name)
        n = len(uni.columns) - len(cols)
        uni.drop(uni.columns[:n], axis=1, inplace=True)
        uni.columns = cols
        uni.reset_index(drop=True, inplace=True)
        tmp.append(uni)

    unigram = pd.concat(tmp)
    unigram["Target"] = np.ones(unigram.shape[0], dtype=np.int)
    os.chdir("../my_data")
    my_df = pd.read_csv("unigrams.csv")
    my_df.drop(my_df.columns[:1], inplace=True, axis=1)
    my_df["Target"] = np.zeros(my_df.shape[0], dtype=np.int)
    unigram = pd.concat([my_df, unigram])
    unigram.index = np.arange(1, unigram.shape[0] + 1)

    print("<-------------------------------------------------UNIGRAM------------------------------------------------->")
    print(unigram)

    return unigram


def get_bigram():
    tmp = []
    os.chdir("/home/bohdan/samsung/2020-knu-ka/classifiers/team_data")

    col = ['.', 't', 'i', 'e', '5', 'Key.shift', 'R', 'o', 'a', 'n', 'l', 'Key.enter']
    cols = []

    for i in range(len(col) - 1):
        cols.append("{} | {}".format(col[i], col[i + 1]))

    for file_name in os.listdir("bigrams"):
        temp = pd.read_csv("bigrams/" + file_name)
        n = len(temp.columns) - len(cols)
        temp.drop(temp.columns[:n], axis=1, inplace=True)
        temp.columns = cols
        temp.reset_index(drop=True, inplace=True)
        tmp.append(temp)

    bigram = pd.concat(tmp)
    bigram["Target"] = np.ones(bigram.shape[0], dtype=np.int)
    os.chdir("../my_data")
    my_df = pd.read_csv("bigrams.csv")
    my_df.drop(my_df.columns[:1], inplace=True, axis=1)
    my_df["Target"] = np.zeros(my_df.shape[0], dtype=np.int)
    bigram = pd.concat([my_df, bigram])
    bigram.index = np.arange(1, bigram.shape[0] + 1)

    print("<-------------------------------------------------BIGRAM------------------------------------------------->")
    print(bigram)

    return bigram


def get_trigram():
    tmp = []
    os.chdir("/home/bohdan/samsung/2020-knu-ka/classifiers/team_data")

    col = ['.', 't', 'i', 'e', '5', 'Key.shift', 'R', 'o', 'a', 'n', 'l', 'Key.enter']
    cols = []

    for i in range(len(col) - 2):
        cols.append("{} | {}".format(col[i], col[i + 2]))

    for file_name in os.listdir("trigrams"):
        temp = pd.read_csv("trigrams/" + file_name)
        n = len(temp.columns) - len(cols)
        temp.drop(temp.columns[:n], axis=1, inplace=True)
        temp.columns = cols
        temp.reset_index(drop=True, inplace=True)
        tmp.append(temp)

    trigram = pd.concat(tmp)
    trigram["Target"] = np.ones(trigram.shape[0], dtype=np.int)
    os.chdir("../my_data")
    my_df = pd.read_csv("trigrams.csv")
    my_df.drop(my_df.columns[:1], inplace=True, axis=1)
    my_df["Target"] = np.zeros(my_df.shape[0], dtype=np.int)
    trigram = pd.concat([my_df, trigram])
    trigram.index = np.arange(1, trigram.shape[0] + 1)

    print("<-------------------------------------------------trigram------------------------------------------------->")
    print(trigram)

    return trigram


def knn(data):
    y = data["Target"]
    x = data.drop(["Target"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    k_range = range(1, 11)
    scores = {}
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores_list)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Testing Accuracy")
    plt.show()


def mlp(data):
    y = data["Target"]
    x = data.drop(["Target"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                        random_state=1)
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)
    print("Accuracy of MLPClassifier :", metrics.accuracy_score(y_test, y_pred))


def rfc(data):
    y = data["Target"]
    x = data.drop(["Target"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    rfc = RandomForestClassifier(n_estimators=20, random_state=0)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    print("Accuracy of Random Forest Classifier :", metrics.accuracy_score(y_test, y_pred))


def linear_regression(data):
    y = data["Target"]
    x = data.drop(["Target"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    print("Accuracy of Linear Regression :", regr.score(x_test, y_test))


def logistic_regression(data):
    y = data["Target"]
    x = data.drop(["Target"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    regr = LogisticRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)
    print("Accuracy of Logistic Regression :", metrics.accuracy_score(y_test, y_pred))


unigram = get_unigram()
bigram = get_bigram()
trigram = get_trigram()

logistic_regression(unigram)
logistic_regression(bigram)
logistic_regression(trigram)

knn(unigram)
knn(bigram)
knn(trigram)

mlp(unigram)
mlp(bigram)
mlp(trigram)

linear_regression(unigram)
linear_regression(bigram)
linear_regression(trigram)

rfc(unigram)
rfc(bigram)
rfc(trigram)
