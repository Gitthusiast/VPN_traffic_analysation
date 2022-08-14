# Feature Selection with Univariate Statistical Tests
import pandas as pd
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import numpy as np


# select best features from all features using ANOVA (f_classif())
def univariate_stat(df, names, no_of_best):
    print("##############################")
    print("########## f_classif #########")
    print("##############################")

    # considering the last column as class labels
    X = df.drop(columns=['categories']).values
    Y = df['categories'].values

    stat_list = [f_classif, mutual_info_classif]

    for stat_test in stat_list:
        # feature extraction
        test = SelectKBest(score_func=stat_test, k=no_of_best)
        fit = test.fit(X, Y)

        # summarize scores
        set_printoptions(precision=3)
        # print(fit.scores_)

        score = {}

        for i, j in zip(names, list(fit.scores_)):
            score[i] = j

        feature_scores = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
        # print(feature_scores)
        print("")
        print("{:<15} {:10}".format('Feature', 'Score'))
        for k, v in feature_scores.items():
            print("{:<15} {:<10}".format(k, v))


# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def recursive_feature_eliminate(df, names, no_of_best):
    print("##############################")
    print("############# RFE ############")
    print("##############################")

    # considering the last column as class labels
    X = df.drop(columns=['categories']).values
    Y = df['categories'].values

    # feature extraction
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model, n_features_to_select=no_of_best)
    fit = rfe.fit(X, Y)

    # print("Num Features: %d" % fit.n_features_)
    # print("Selected Features: %s" % fit.support_)
    # print("Feature Ranking: %s" % fit.ranking_)

    selection = {}
    for i, j in zip(names, list(fit.ranking_)):
        selection[i] = j

    support = {}
    for i, j in zip(names, list(fit.support_)):
        support[i] = j
    # print(support)
    print("{:<15} {:<10}".format('Feature', 'Support'))
    for k, v in support.items():
        print("{:<15} {:<10}".format(k, v))

    feature_rank = dict(sorted(selection.items(), key=lambda item: item[1]))
    # print(feature_rank)
    print("")
    print("{:<15} {:<10}".format('Feature', 'Rank'))
    for k, v in feature_rank.items():
        print("{:<15} {:<10}".format(k, v))


# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier


def extra_tree_classifier(df, names):
    print("##############################")
    print("#### ExtraTreesClassifier ####")
    print("##############################")

    # considering the last column as class labels
    X = df.drop(columns=['categories']).values
    Y = df['categories'].values

    # feature extraction
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, Y)
    # print(model.feature_importances_)

    importance = {}

    for i, j in zip(names, list(model.feature_importances_)):
        importance[i] = j

    feature_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    # print(feature_importance)
    print("{:<15} {:<10}".format('Feature', 'Importance'))
    for k, v in feature_importance.items():
        print("{:<15} {:<10}".format(k, v))


# import VarianceThreshold
from sklearn.feature_selection import VarianceThreshold

if __name__ == "__main__":
    filename = ".\\FinalLabeled10KEachEngland.tsv"
    # filename = ".\\FinalLabeled10KEachJapan.tsv.tsv"
    names = ['frame.time_epoch', 'frame.len', 'tcp.srcport', 'categories', 'io_packet', 'time_delta',
             'average_delta_time', 'std_delta_time', 'average_len', 'std_len']

    df = read_csv(filename, usecols=names, sep='\t')

    # df['os'] = df['os'].replace(df.os.unique(), [i for i in range(len(df.os.unique()))])
    df = df.loc[:, (df != df.iloc[0]).any()]
    names = list(df.columns)
    df.to_csv(".\\FinalLabeled10KEachEngland_Train.tsv", index=False, sep='\t')
    print(names)

    # no_of_best = int(input("Enter the no. of best features: "))
    no_of_best = 9

    names_no_cet = ['frame.time_epoch', 'frame.len', 'tcp.srcport', 'io_packet', 'time_delta',
             'average_delta_time', 'std_delta_time', 'average_len', 'std_len']
    print("")
    univariate_stat(df, names_no_cet, no_of_best)
    print("")
    recursive_feature_eliminate(df, names_no_cet, no_of_best)
    print("")
    extra_tree_classifier(df, names_no_cet)