import pandas as pd
import numpy as np


def unigram(filename):
    data = []
    col = ["H.", "Ht", "Hi", "He", "H5", "Hshift", "Hr", "Ho", "Ha", "Hn", "Hl", "Henter"]

    k = pd.read_csv(filename)
    for _, row in k.iterrows():
        data_row = []
        for i in col:
            data_row.append(row[i])
        data.append(data_row)

    df = pd.DataFrame(np.array(data), columns=["H.", "Ht", "Hi", "He", "H5", "Hshift", "Hr", "Ho", "Ha", "Hn", "Hl",
                                               "Henter"])
    df.to_csv("unigrams.csv", encoding='utf-8')


def bigram(filename):
    data = []

    k = pd.read_csv(filename,
                    usecols=["DD.", "Ht", "DDt", "Hi", "DDi", "He", "DDe", "H5", "DD5", "Hshift", "DDshift",
                             "Hr", "DDr", "Ho", "DDo", "Ha", "DDa", "Hn", "DDn", "Hl", "DDl", "Henter"])
    for i, row in k.iterrows():
        arr = row.array
        r = []
        for j in range(1, len(arr), 2):
            r.append(arr[j] + arr[j-1])
        data.append(r)

    df = pd.DataFrame(data, columns=[".-t", "t-i", "i-e", "e-5", "5-shift", "shift-r", "r-o", "o-a", "a-n", "n-l",
                                     "l-enter"])
    df.to_csv("bigrams.csv", encoding='utf-8')


def trigram(filename):
    data = []

    k = pd.read_csv(filename,
                    usecols=["H.", "UD.", "Ht", "UDt", "Hi", "UDi", "He", "UDe", "H5", "UD5", "Hshift", "UDshift",
                             "Hr", "UDr", "Ho", "UDo", "Ha", "UDa", "Hn", "UDn", "Hl", "UDl", "Henter"])
    for i, row in k.iterrows():
        arr = row.array
        r = []
        for j in range(4, len(arr), 2):
            r.append(arr[j] + arr[j-1] + arr[j-2] + arr[j-3] + arr[j-4])
        data.append(r)

    df = pd.DataFrame(data, columns=[".-i", "t-e", "i-5", "e-shift", "5-r", "shift-o", "r-a", "o-n", "a-l", "n-enter"])
    df.to_csv("trigrams.csv", encoding='utf-8')


unigram("dataset.csv")
bigram("dataset.csv")
trigram("dataset.csv")
