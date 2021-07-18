from load_data import load_data
from utils import compute_error
from utils import get_data_grid
from utils import reshape_grid
from utils import get_data_wave
from cnn import cnn_training_schemes

import numpy as np
import keras
import os
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score


if __name__ == "__main__":

    averaging_at_inference_time = True
    stochastic_averaging = False

    eval_mode = False

    num_sampled_points = 200
    dim_of_interest = 0

    train_x, test_x, train_y = load_data()

    if os.path.isfile("train_x_grid.p"):
        train_x = pickle.load(open("train_x_grid.p", "rb"))
        test_x = pickle.load(open("test_x_grid.p", "rb"))

    else:
        train_x = get_data_grid(train_x)
        train_x = reshape_grid(train_x)

        test_x = get_data_grid(test_x)
        test_x = reshape_grid(test_x)

        pickle.dump(train_x, open("train_x_grid.p", "wb"))
        pickle.dump(test_x, open("test_x_grid.p", "wb"))
        pickle.dump(train_y, open("train_y.p", "wb"))

    if eval_mode:

        dim_of_interest = 0
        y_bins = np.digitize(train_y[:, dim_of_interest], np.linspace(np.min(train_y[:, dim_of_interest]), np.max(train_y[:, dim_of_interest]), 3))

        # nested cross-validation
        list_indices_outer = []
        list_indices_inner = []  # this is used for the validation

        list_val_loss = []
        list_val_r2 = []

        num_eval = 0
        kf_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        for train_outer, val_outer in kf_outer.split(train_x, y_bins):
            list_indices_outer.append((train_outer, val_outer))

            x_train_outer, y_train_outer, y_bins_train = train_x[train_outer], train_y[train_outer], y_bins[train_outer]
            x_val_outer, y_val_outer = train_x[val_outer], train_y[val_outer]

            kf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            for train_inner, val_inner in kf_inner.split(x_train_outer, y_bins_train):
                list_indices_inner.append((train_inner, val_outer))

                x_train_inner, y_train_inner = x_train_outer[train_inner], y_train_outer[train_inner]
                x_val_inner, y_val_inner = x_train_outer[val_inner], y_train_outer[val_inner]

                predictions_train, predictions_val = cnn_training_schemes(x_train_inner, x_val_inner, y_train_inner, y_val_inner, dim_of_interest,
                                                                              num_eval, stochastic_averaging, averaging_at_inference_time)

                train_error = compute_error(y_train_inner[:, dim_of_interest], predictions_train)
                error = compute_error(y_val_inner[:, dim_of_interest], predictions_val)
                r2 = r2_score(y_val_inner[:, dim_of_interest], predictions_val)
                list_val_loss.append(error)
                list_val_r2.append(r2)

                print("TRAIN ERROR", train_error, "VAL ERROR: ", error, "R^2: ", r2)
                print("AVERAGE ERROR: ", np.mean(list_val_loss), "AVERAGE R2: ", np.mean(list_val_r2))

                num_eval += 1

                if num_eval % 10 == 0:
                    print("the number of eval is:" + str(num_eval))
                    break

    else:

        test_pred = [[] for _ in range(2)]
        index = 0
        num_eval = 0
        for dim_of_interest in [0, 2]:
            r2_list = []
            y_bins = np.digitize(train_y[:, dim_of_interest], np.linspace(np.min(train_y[:, dim_of_interest]), np.max(train_y[:, dim_of_interest]), 3))

            kf_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            for train_outer, val_outer in kf_outer.split(train_x, y_bins):

                x_train_outer, y_train_outer = train_x[train_outer], train_y[train_outer]
                x_val_outer, y_val_outer = train_x[val_outer], train_y[val_outer]

                predictions_train, predictions_val, predictions_test = cnn_training_schemes(x_train_outer, x_val_outer, y_train_outer,
                                                                                                y_val_outer, test_x,
                                                                                                dim_of_interest, num_eval, stochastic_averaging,
                                                                                                averaging_at_inference_time)

                test_pred[index].append(predictions_test)

                # plt.plot()
                # plt.grid(True)
                # plt.scatter(y_val_outer[:, dim_of_interest], predictions_val, s=5)
                # plt.show()

                train_error = compute_error(y_train_outer[:, dim_of_interest], predictions_train)
                error = compute_error(y_val_outer[:, dim_of_interest], predictions_val)
                train_r2 = r2_score(y_train_outer[:, dim_of_interest], predictions_train)
                r2 = r2_score(y_val_outer[:, dim_of_interest], predictions_val)
                r2_list.append(r2)

                print("TRAIN ERROR: ", train_error, "TRAIN R2: ", train_r2, "VAL ERROR: ", error, "VAL R2: ", r2)
                num_eval += 1

                keras.backend.clear_session()

            best_model_index = np.argmax(r2_list)
            test_pred[index] = test_pred[index][best_model_index]

            index += 1

    pickle.dump(test_pred, open("results.p", "wb"))
