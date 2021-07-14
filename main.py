from utils import get_one_dim_diff
from load_data import load_data
from utils import compute_error
from utils import clip_values
from cnn import keras_model
from cnn import cnn_training_schemes

import numpy as np
import keras
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score


def get_data(data_x, binned=False, num_sampled_points=200, n_bins=20):

    length_multi_instance = num_sampled_points

    diff_data = np.zeros((len(data_x), length_multi_instance))

    for index in range(len(data_x)):

        if index % 10 == 0:
            print("COMPUTED FOR INDEX NUMBER: " + str(index))

        pred_before, pred_after, range_pred = get_one_dim_diff(data_x[index][0], data_x[index][1], num_sampled_points)
        feature = pred_before - pred_after

        diff_data[index, :] = feature

    if binned:
        final_data = np.zeros((len(data_x), n_bins))
        min_val, max_val = np.min(diff_data), np.max(diff_data)

        for index in range(diff_data.shape[0]):
            final_data[index, :] = np.histogram(diff_data[index], n_bins, (min_val, max_val), density=False)[0]/num_sampled_points

        return final_data

    else:

        return diff_data


if __name__ == "__main__":

    binned = True

    averaging_at_inference_time = False
    stochastic_averaging = False

    eval_mode = True

    num_sampled_points = 200
    dim_of_interest = 0

    train_x, test_x, train_y = load_data()
    train_x = get_data(train_x, binned, num_sampled_points)
    test_x = get_data(test_x, binned, num_sampled_points)

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

                if binned:
                    model = GradientBoostingRegressor()
                    model.fit(x_train_inner, y_train_inner[:, dim_of_interest])
                    predictions_train = model.predict(x_train_inner)
                    predictions_val = model.predict(x_val_inner)

                else:
                    predictions_train, predictions_val = cnn_training_schemes(x_train_inner, x_val_inner, y_train_inner, y_val_inner, dim_of_interest,
                                                                              num_eval, stochastic_averaging, averaging_at_inference_time)

                train_error = compute_error(y_train_inner[:, dim_of_interest], predictions_train)
                error = compute_error(y_val_inner[:, dim_of_interest], predictions_val)
                r2 = r2_score(y_val_inner[:, dim_of_interest], predictions_val)
                list_val_loss.append(error)
                list_val_r2.append(r2)

                print("TRAIN ERROR", train_error, "VAL ERROR: ", error, "R^2: ", r2)
                print("AVERAGE ERROR: ", np.mean(list_val_loss), "AVERAGE R2: ", np.mean(list_val_r2))

                # plt.plot()
                # plt.grid(True)
                # plt.scatter(y_val_inner[:, dim_of_interest], predictions_val, s=5)
                # plt.show()

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

                x_train_outer, y_train_outer, y_bins_train = train_x[train_outer], train_y[train_outer], y_bins[train_outer]
                x_val_outer, y_val_outer = train_x[val_outer], train_y[val_outer]

                x_train_outer, x_val_outer = x_train_outer / np.abs(np.max(x_train_outer)), x_val_outer / np.abs(np.max(x_train_outer))
                test_x = test_x / np.abs(np.max(x_train_outer))
                model = keras_model(num_eval, False)

                early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

                model.fit(x_train_outer, y_train_outer[:, dim_of_interest], batch_size=32, epochs=200,
                            verbose=2, validation_data=(x_val_outer, y_val_outer[:, dim_of_interest]), callbacks=[early_stop_callback])

                weights_trained = model.get_weights()
                model = keras_model(num_eval, True)
                model.set_weights(weights_trained)

                list_pred_val = []
                list_pred_test = []

                for _ in range(200):
                    predictions_val = model.predict(x_val_outer)
                    predictions_val = clip_values(predictions_val, dim_of_interest)

                    predictions_test = model.predict(test_x)
                    predictions_test = clip_values(predictions_test, dim_of_interest)

                    list_pred_val.append(predictions_val)
                    list_pred_test.append(predictions_test)

                predictions_val = np.mean(np.array(list_pred_val), axis=0)
                predictions_test = np.mean(np.array(list_pred_test), axis=0)

                plt.plot()
                plt.grid(True)
                plt.scatter(y_val_outer[:, dim_of_interest], predictions_val, s=5)
                plt.show()

                test_pred[index].append(predictions_test)

                error = compute_error(y_val_outer[:, dim_of_interest], predictions_val)
                r2 = r2_score(y_val_outer[:, dim_of_interest], predictions_val)
                r2_list.append(r2)

                print("ERROR: ", error, " R2: ", r2)
                num_eval += 1

                keras.backend.clear_session()

            best_model_index = np.argmax(r2_list)
            test_pred[index] = test_pred[index][best_model_index]

            index += 1

    pickle.dump(test_pred, open("results.p", "wb"))




