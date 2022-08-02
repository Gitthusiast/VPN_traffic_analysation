
from ML_Machine_Learning import ClassifierModel
import pandas as pd

if __name__ == "__main__":

    filename = ".\\FinalLabeled10KEachEngland.tsv"

    names = ['frame.len', 'tcp.srcport', 'Random website', 'Browsing', 'Country', 'in/out', 'time_delta',
             'average_delta_time', 'std_delta_time', 'average_len', 'std_len']

    labeled_df = pd.read_csv(filename, usecols=names, sep='\t')
    print(labeled_df)

    x_iloc_list = [0, 1, 5, 7, 8, 8, 10, 11]  # indexes in the labeled csv
    y_iloc = 4

    model = ClassifierModel(names, labeled_df, x_iloc_list, y_iloc)

    model.KNN()
    model.NB()
    model.ANN()
    model.RF()
    model.DT()
    model.SVM('rbf')
    model.SVM('linear')
    model.XGBOOST()

    model.run_models()
