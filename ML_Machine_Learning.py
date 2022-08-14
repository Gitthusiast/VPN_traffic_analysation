import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
import joblib
from os import listdir
from os.path import isfile
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
import xgboost as xgb

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer



class ClassifierModel:

    def __init__(self, feature_names, dataset, x_iloc_list, y_iloc, testSize=0.2):

        # From dataset:
        X = dataset.iloc[:, x_iloc_list].values
        y = dataset.iloc[:, y_iloc].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=0)

        self.test_set = pd.read_csv("FinalLabeledFileTransferJapan.tsv", usecols=feature_names, sep='\t')
        self.feature_names = feature_names
        self.x_iloc_list = x_iloc_list
        self.y_iloc = y_iloc
        sc = StandardScaler()
        self.sc = sc
        self.x_train = sc.fit_transform(X_train)
        self.y_train = y_train
        self.x_test = sc.transform(X_test)
        self.y_test = y_test
        self.models_accuracy = []

    # ****************** Scores: ************************************

    def accuracy(self, confusion_matrix):

        sum, total = 0, 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                if i == j:
                    sum += confusion_matrix[i, j]
                total += confusion_matrix[i, j]
        return sum/total

    def classification_report_plot(self, clf_report, filename, folder="england_cm_plots"):

        if not os.path.isdir(folder):
            os.mkdir(folder)

        out_file_name = folder + "/" + filename + ".png"

        fig = plt.figure(figsize=(16, 10))
        sns.set(font_scale=4)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Blues")
        fig.savefig(out_file_name, bbox_inches="tight")

    def confusion_matrix_report_plot(self, loaded_model, filename, X, Y, experiment_alias=''):

        folder = "confusion_matrix"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        if experiment_alias != '':
            folder = folder + '/' + experiment_alias
            if not os.path.isdir(folder):
                os.mkdir(folder)

        out_file_name = folder + "/" + filename + ".png"
        matrix = plot_confusion_matrix(loaded_model, self.x_test, self.y_test,
                                       labels=['Chat', 'Streaming', 'Random_Websites',
                                               'File Transfer', 'Video Conferencing'],
                                       values_format='.2%', normalize='true',  cmap=plt.cm.Blues)
        title = 'Confusion Matrix ' + filename.upper() + '\n' + experiment_alias
        matrix.ax_.set_title(title, color='Black')
        plt.xlabel('Predicted Label', color='Black')
        plt.ylabel('True Label', color='Black')
        matrix.figure_.savefig(out_file_name, bbox_inches="tight")

    def visualize_feature_importance(self, loaded_model, feature_names, filename):

        importance = []

        if type(loaded_model) == DecisionTreeClassifier:
            importance = loaded_model.feature_importances_
            # summarize feature importance
            for i, v in enumerate(importance):
                print('Feature: %s, Score: %.5f' % (feature_names[i], v))
            # plot feature importance

        elif type(loaded_model) == MLPClassifier:
            importance = loaded_model.coefs_[0][0]
            # summarize feature importance
            for i, v in enumerate(importance):
                print('Feature: %s, Score: %.5f' % (feature_names[i], v))

        elif type(loaded_model) == RandomForestClassifier:
            importance = loaded_model.feature_importances_
            # summarize feature importance
            for i, v in enumerate(importance):
                print('Feature: %s, Score: %.5f' % (feature_names[i], v))
            fig = plt.figure(figsize=(10, 5))
            plt.barh(feature_names, importance, color='#199CEC')
            plt.xlabel("Feature importance")
            plt.ylabel("Feature name")
            plt.title(f"Feature importance")  #RF
            plt.show()

        # elif type(loaded_model) == KNeighborsClassifier:
        #     le_res = self.own_label_encoder(self.Y)
        #     results = permutation_importance(loaded_model, self.X, le_res,
        #                                      scoring='neg_mean_squared_error')
        #     # get importance
        #     importance = results.importances_mean
        #     # summarize feature importance

    def k_fold(self, estimator, k, estimator_name):

        kfold = KFold(n_splits=k, shuffle=True, random_state=np.random.seed(7))
        results = cross_val_score(estimator, self.X, self.Y, cv=kfold)
        print(f"****{estimator_name}:****")
        self.models_accuracy.append((results.mean(), results.std()))
        print("Baseline accuracy: (%.2f%%) with std: (%.2f%%)" % (results.mean()*100, results.std()*100))

    # ****************** MODELS: ************************************

    def ANN(self):

        ANN_Classifier = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(100, 100),
                                       max_iter=400, tol=1e-6, random_state=1)
        ANN_Classifier.fit(self.x_train, self.y_train)
        joblib.dump(ANN_Classifier, "model/ann.sav")
        y_pred = ANN_Classifier.predict(self.x_test)

        self.classification_report_plot(classification_report(self.y_test, y_pred,
                                                              output_dict=True), "ANN")

        print("\n")
        print("************************* Nueral Network Classifier ************************* \n")

    def SVM(self, kernel_type="linear"):
        SVM_Classifier = SVC()
        SVM_Classifier.fit(self.x_train, self.y_train)
        joblib.dump(SVM_Classifier, "model/svm" + kernel_type + '.sav')
        y_pred = SVM_Classifier.predict(self.x_test)

        self.classification_report_plot(classification_report(self.y_test, y_pred,
                                                              output_dict=True), f"SVM-{kernel_type}")

        print("\n")
        print("*************************Support Vector Classifier************************* \n")

    def RF(self):
        RF_Classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, max_depth=5,
                                               min_samples_leaf=100)
        RF_Classifier.fit(self.x_train, self.y_train)
        joblib.dump(RF_Classifier, "model/rf.sav")
        y_pred = RF_Classifier.predict(self.x_test)

        self.classification_report_plot(classification_report(self.y_test, y_pred,
                                                              output_dict=True), "RF")

        print("\n")
        print("************************* Random Forest Classifier ************************* \n")

    def NB(self):
        NB_Classifier = GaussianNB(var_smoothing=1e-3)

        NB_Classifier.fit(self.x_train, self.y_train)
        joblib.dump(NB_Classifier, "model/nb.sav")
        y_pred = NB_Classifier.predict(self.x_test)

        self.classification_report_plot(classification_report(self.y_test, y_pred,
                                                              output_dict=True), "NB")

        print("\n")
        print("************************* Naive Bayes Classifier *************************\n")

    def KNN(self):
        KNN_Classifier = KNeighborsClassifier()
        KNN_Classifier.fit(self.x_train, self.y_train)
        joblib.dump(KNN_Classifier, "model/knn.sav")
        y_pred = KNN_Classifier.predict(self.x_test)

        self.classification_report_plot(classification_report(self.y_test, y_pred,
                                                              output_dict=True), "KNN")

        print("\n")
        print("************************* K-Neighbors Classifier *************************\n")

    def XGBOOST(self):

        xgb_model = xgb.XGBClassifier()
        print("\n")
        print("************************* XGBoost Classifier *************************\n")
        target = self.own_label_encoder(self.y_train)
        xgb_model.fit(self.x_train, target)
        joblib.dump(xgb_model, "model/xgb.sav")
        y_pred = xgb_model.predict(self.x_test)
        y_pred = self.own_inverse_label_encoder(y_pred)

        self.classification_report_plot(classification_report(self.y_test, y_pred,
                                                              output_dict=True), "XGBOOST")

    def DT(self):
        DT_Classifier = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=5, min_samples_leaf=10)
        DT_Classifier.fit(self.x_train, self.y_train)
        joblib.dump(DT_Classifier, "model/dt.sav")
        y_pred = DT_Classifier.predict(self.x_test)

        self.classification_report_plot(classification_report(self.y_test, y_pred,
                                                              output_dict=True), "DT")

        print("\n")
        print("************************* Decision Tree Classifier *************************\n")

    def models_summery(self):
        folder = "clf_plots_monday"
        if not os.path.isdir(folder):
            os.mkdir(folder)
        out_file_name = folder + "/summary.png"
        accuracies = pd.DataFrame(
            self.models_accuracy, columns=['Accuracy', 'Std'],
            # index=['KNN', 'linearSVM', 'rbfSVM', 'NB', 'RF', 'ANN', 'DT'])
            index=['KNN', 'NB', 'RF', 'ANN', 'DT', 'XGBOOST'])
        fig = plt.figure(figsize=(16, 10))
        sns.set(font_scale=4)
        sns.heatmap(accuracies, annot=True, cmap="BuPu")
        fig.savefig(out_file_name, bbox_inches="tight")

    def own_inverse_label_encoder(self, to_transform):
        """dict_transform_lables = {0: 'Browsing', 1: 'Chat', 2: 'Streaming', 3: 'File Transfer',
                                 4: 'Video Conferencing'}"""
        # dict_transform_lables = {0: 'Youtube', 1: 'Netflix', 2: 'Vimeo'}
        # dict_transform_lables = {0: 'Skype_chat', 1: 'Facebook', 2: 'GoogleHangouts',
        #                          3: 'Whatsapp_chat', 4: 'Telegram_chat'}
        # dict_transform_lables = {0: 'Skype_video', 1: 'GoogleMeets', 2: 'Zoom',
        #                          3: 'Microsoft_teams'}
        # dict_transform_lables = {0: 'Skype_video', 1: 'GoogleMeets', 2: 'Zoom',
        #                          3: 'Microsoft_teams'}
        dict_transform_lables = {0: 'qBitTorrent', 1: 'Skype_files', 2: 'Dropbox',
                                 3: 'gdrive', 4: 'Whatsapp_files', 5: 'Telegram_files'}
        target = to_transform
        results = []
        for label in target:
            results.append(dict_transform_lables[label])
        return results

    def own_label_encoder(self, to_transform):
        """dict_transform_lables = {'Browsing': 0, 'Chat': 1, 'Streaming': 2, 'File Transfer': 3,
                                 'Video Conferencing': 4}"""
        # dict_transform_lables = {'Youtube': 0, 'Netflix': 1, 'Vimeo': 2}
        # dict_transform_lables = {'Skype_chat': 0, 'Facebook': 1, 'GoogleHangouts': 2,
        #                          'Whatsapp_chat': 3, 'Telegram_chat': 4}
        # dict_transform_lables = {'qBitTorrent': 0, 'qBittorrent': 1, 'Skype_files': 2,
        #                          'Dropbox': 3, 'gdrive': 4, 'Whatsapp_files': 5, 'Telegram_files': 6}
        dict_transform_lables = {'qBitTorrent': 0, 'Skype_files': 1, 'Dropbox': 2,
                                 'gdrive': 3, 'Whatsapp_files': 4, 'Telegram_files': 5}
        target = []
        for label in to_transform:
            target.append(dict_transform_lables[label])
        return target

    def run_models(self):
        """
        This funciton returns the final classification results on Japan
        :return:
        """

        X = self.test_set.iloc[:, self.x_iloc_list].values
        X = self.sc.fit_transform(X)
        Y = self.test_set.iloc[:, self.y_iloc].values
        models = ["model/" + f for f in listdir("model") if isfile("model/" + f)]
        for filename in models:
            print("******************\n" + filename + "\n******************\n")
            loaded_model = joblib.load(filename)

            filename = filename.split('/')[1].split('.')[0]
            results = loaded_model.predict(X)

            if filename == 'xgb':
                results = self.own_inverse_label_encoder(results)

            self.classification_report_plot(classification_report(Y, results, output_dict=True),
                                            filename, "japan_cm_plots")

            # self.visualize_feature_importance(loaded_model, self.feature_names[:-1], filename)

            cm = confusion_matrix(Y, results)*100
            print('Precision: ', self.accuracy(cm), '%')
            print("******************\n")
            # self.confusion_matrix_report_plot(loaded_model, filename, X, Y, "browsing_classification")

            result = loaded_model.score(X, Y)
            print(result)
            print("******************\n")


