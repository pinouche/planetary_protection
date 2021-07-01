from utils import get_one_dim_diff
from load_data import load_data
from utils import compute_error
from utils import clip_values
from cnn import keras_model

import numpy as np
import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score


def get_data(train_x):

    final_data = np.zeros((200, 200))

    for index in range(len(train_x)):

        if index % 10 == 0:
            print("COMPUTED FOR INDEX NUMBER: " + str(index))

        # feature = make_grid(train_x[index][0], train_x[index][1],
        #                            np.linspace(0, 5000, 20),
        #                            np.array([0.92, 0.94, 0.96, 0.98, 1]))

        pred_before, pred_after, range_pred = get_one_dim_diff(train_x[index][0],
                                                               train_x[index][1],
                                                               200)
        feature = pred_before - pred_after

        final_data[index, :] = feature

    return final_data


if __name__ == "__main__":

    train_x, test_x, train_y = load_data()
    final_data = get_data(train_x)

    dim_of_interest = 0
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

            # x_train_inner, x_val_inner = standardize(x_train_inner, x_val_inner, True)
            x_train_inner, x_val_inner = x_train_inner / np.abs(np.max(x_train_inner)), x_val_inner / np.abs(np.max(x_train_inner))
            model = keras_model(num_eval, False)
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                                     restore_best_weights=True)
            model.fit(x_train_inner, y_train_inner[:, dim_of_interest], batch_size=8, epochs=300,
                      verbose=2, validation_data=(x_val_inner, y_val_inner[:, dim_of_interest]),
                      callbacks=[callback])

            # predictions_val = model.predict(x_val_inner)
            # predictions_val = clip_values(predictions_val, dim_of_interest)

            weights_trained = model.get_weights()
            model = keras_model(num_eval, True)
            model.set_weights(weights_trained)

            list_pred = []
            for _ in range(100):
                predictions_val = model.predict(x_val_inner)
                predictions_val = clip_values(predictions_val, dim_of_interest)
                list_pred.append(predictions_val)

            predictions_val = np.mean(np.array(list_pred), axis=0)

            error = compute_error(y_val_inner[:, dim_of_interest], predictions_val)
            r2 = r2_score(y_val_inner[:, dim_of_interest], predictions_val)
            list_val_loss.append(error)
            list_val_r2.append(r2)
            print("ERROR: ", error, "R^2: ", r2)

            # plt.plot()
            # plt.grid(True)
            # plt.scatter(y_val_inner[:, 2], predictions_val)
            # plt.show()

            num_eval += 1

            if num_eval % 10 == 0:
                print("the number of eval is:" + str(num_eval))
                break

            keras.backend.clear_session()






