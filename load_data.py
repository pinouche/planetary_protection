import pandas as pd
import numpy as np
import os
import re


def load_data():
    size_train_set, size_test_set = 200, 100
    train_x, test_x = [[] for _ in range(size_train_set)], [[] for _ in range(size_test_set)]

    path = "lightcurves"
    file_list = [file for file in os.listdir(path)]
    file_list.sort()

    for file in file_list:
        data = pd.read_csv(path + "/" + file, sep=",", header=None)

        if file == "parameters.csv":
            train_y = np.array(np.array(data))
        else:
            file_number = int(re.findall(r'\d+', file)[0])

            if file_number <= size_train_set:
                train_x[file_number - 1].append(np.array(data))
            else:
                test_x[file_number - size_train_set - 1].append(np.array(data))

    return train_x, test_x, train_y