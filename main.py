import dataEngineering as dataE
from ML_Machine_Learning import ClassifierModel
import pandas as pd

if __name__ == '__main__':

    # dataE.data_eng()

    filename = ".\\FinalLabeled10KEachEngland.tsv"

    names = ['frame.len', 'tcp.srcport', 'io_packet',  'average_delta_time', 'std_delta_time', 'average_len',
             'std_len', 'time_delta', 'categories']

    labeled_df = pd.read_csv(filename, usecols=names, sep='\t')
    print(labeled_df)

    x_iloc_list = [0, 1, 3, 4, 5, 6, 7, 8]  # indexes in the labeled csv
    y_iloc = 2

    model = ClassifierModel(names, labeled_df, x_iloc_list, y_iloc)

    model.KNN()
    model.NB()
    model.ANN()
    model.RF()
    model.DT()
    model.XGBOOST()
    # model.SVM('linear')

    model.run_models()
