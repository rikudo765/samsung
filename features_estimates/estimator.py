import pandas as pd
from scipy.stats import kstest, chisquare, skew, kurtosis


def unigram_estimates(filenames):
    for filename in filenames:
        data = pd.read_csv(filename)
        estimates = []
        for col in data.columns[1:]:
            ks = kstest(data[col], "norm")
            x2 = chisquare(data[col])
            estimates.append([col, data[col].mean(), data[col].var(), data[col].std(), skew(data[col]), kurtosis(data[col]),
                              "stat={}, pvalue={}".format(ks[0], ks[1]),
                              "stat={}, pvalue={}".format(x2[0], x2[1])])

        df = pd.DataFrame(estimates, columns=["Key", "Mean", "Variance", "Standard deviation", "Skewness", "Kurtosis",
                                              "K-S test", "X^2 test"])
        df.to_csv("{}_estimates.csv".format(filename[:filename.find("s")]))


unigram_estimates(["unigrams.csv", "bigrams.csv", "trigrams.csv"])
