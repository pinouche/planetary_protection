from utils import get_one_dim_diff
from load_data import load_data
from utils import compute_error
from utils import clip_values
from cnn import keras_model

import numpy as np
import keras
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score


def get_data(train_x, num_intervals=4, num_sampled_points=200):

    length_multi_instance = int(num_sampled_points / num_intervals)
    final_data = np.zeros((200 * num_intervals, length_multi_instance))

    counter = 0
    for index in range(len(train_x)):

        if index % 10 == 0:
            print("COMPUTED FOR INDEX NUMBER: " + str(index))

        pred_before, pred_after, range_pred = get_one_dim_diff(train_x[index][0], train_x[index][1], num_sampled_points)
        feature = pred_before - pred_after

        for n_interval in range(num_intervals):
            n_f = feature[length_multi_instance * n_interval:length_multi_instance * (n_interval + 1)]

            final_data[counter, :] = n_f

            counter += 1

    return final_data


if __name__ == "__main__":

    averaging_at_inference_time = False
    stochastic_averaging = True

    num_intervals = 1
    num_sampled_points = 200
    dim_of_interest = 0

    train_x, test_x, train_y = load_data()
    final_data = get_data(train_x, num_intervals, num_sampled_points)

    train_y = np.repeat(train_y, num_intervals, axis=0)
    y_bins = np.digitize(train_y[:, dim_of_interest], np.linspace(np.min(train_y[:, dim_of_interest]), np.max(train_y[:, dim_of_interest]), 4))

    # nested cross-validation
    list_indices_outer = []
    list_indices_inner = []  # this is used for the validation

    list_val_loss = []
    list_val_r2 = []

    num_eval = 0
    kf_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_outer, val_outer in kf_outer.split(final_data, y_bins):
        list_indices_outer.append((train_outer, val_outer))

        x_train_outer, y_train_outer, y_bins_train = final_data[train_outer], train_y[train_outer], y_bins[train_outer]
        x_val_outer, y_val_outer = final_data[val_outer], train_y[val_outer]

        kf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        for train_inner, val_inner in kf_inner.split(x_train_outer, y_bins_train):
            list_indices_inner.append((train_inner, val_outer))

            x_train_inner, y_train_inner = x_train_outer[train_inner], y_train_outer[train_inner]
            x_val_inner, y_val_inner = x_train_outer[val_inner], y_train_outer[val_inner]

            x_train_inner, x_val_inner = x_train_inner / np.abs(np.max(x_train_inner)), x_val_inner / np.abs(np.max(x_train_inner))
            model = keras_model(num_eval, False)

            early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

            def scheduler(epoch, lr):
                if epoch >= 0:
                    lr = 0.0005
                return lr

            scheduler_callback = keras.callbacks.LearningRateScheduler(scheduler)

            model.fit(x_train_inner, y_train_inner[:, dim_of_interest], batch_size=16, epochs=100,
                      verbose=2, validation_data=(x_val_inner, y_val_inner[:, dim_of_interest]))

            if stochastic_averaging:

                weight_list = []
                for _ in range(20):
                    model.fit(x_train_inner, y_train_inner[:, dim_of_interest], batch_size=16, epochs=1,
                              verbose=2, validation_data=(x_val_inner, y_val_inner[:, dim_of_interest]),
                              callbacks=[scheduler_callback])

                    weights = model.get_weights()
                    weight_list.append(weights)

                weights = np.mean(np.array(weight_list), axis=0)
                model = keras_model(num_eval, False)
                model.set_weights(weights)

            else:
                model.fit(x_train_inner, y_train_inner[:, dim_of_interest], batch_size=16, epochs=200,
                          verbose=2, validation_data=(x_val_inner, y_val_inner[:, dim_of_interest]),
                          callbacks=[early_stop_callback])

            if averaging_at_inference_time:

                weights_trained = model.get_weights()
                model = keras_model(num_eval, True)
                model.set_weights(weights_trained)

                list_pred = []
                list_pred_train = []
                for _ in range(200):
                    predictions_val = model.predict(x_val_inner)
                    predictions_val = clip_values(predictions_val, dim_of_interest)

                    predictions_train = model.predict(x_train_inner)
                    predictions_train = clip_values(predictions_train, dim_of_interest)

                    list_pred.append(predictions_val)
                    list_pred_train.append(predictions_train)

                predictions_val = np.mean(np.array(list_pred), axis=0)
                predictions_train = np.mean(np.array(list_pred_train), axis=0)

            else:
                predictions_val = model.predict(x_val_inner)
                predictions_val = clip_values(predictions_val, dim_of_interest)

                predictions_train = model.predict(x_train_inner)
                predictions_train = clip_values(predictions_train, dim_of_interest)

            train_error = compute_error(y_train_inner[:, dim_of_interest], predictions_train)
            error = compute_error(y_val_inner[:, dim_of_interest], predictions_val)
            r2 = r2_score(y_val_inner[:, dim_of_interest], predictions_val)
            list_val_loss.append(error)
            list_val_r2.append(r2)

            print("TRAIN ERROR", train_error, "VAL ERROR: ", error, "R^2: ", r2)
            print("AVERAGE ERROR: ", np.mean(list_val_loss), "AVERAGE R2: ", np.mean(list_val_r2))

            plt.plot()
            plt.grid(True)
            plt.scatter(y_val_inner[:, dim_of_interest], predictions_val, s=5)
            plt.show()

            num_eval += 1

            if num_eval % 10 == 0:
                print("the number of eval is:" + str(num_eval))
                break

            keras.backend.clear_session()






