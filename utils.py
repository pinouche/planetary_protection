import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def compute_baseline_error(list_indices, train_y, dim):
    list_error = []
    for index in range(len(list_indices)):
        train_target = train_y[list_indices[index][0]][:, dim]
        test_target = train_y[list_indices[index][1]][:, dim]

        predictions_target = np.expand_dims(np.mean(train_target, axis=0), axis=0)
        predictions_target = np.repeat(predictions_target, test_target.shape[0], axis=0)
        loss = compute_error(predictions_target, test_target)
        list_error.append(loss)

    return list_error


def compute_error(true_array, predicted_array):
    loss = np.mean(np.sqrt(np.mean((true_array - predicted_array) ** 2, axis=0)))

    return loss


def scale_data(train_x, val_x):
    scaler = StandardScaler()
    scaler = scaler.fit(train_x)

    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)

    return train_x, val_x


def standardize_grid(train_data, val_data, patch_wise=True):
    if patch_wise:
        train_mean, std_mean = np.mean(train_data, axis=0), np.std(train_data, axis=0) + (10e-4)
    else:
        train_mean, std_mean = np.mean(train_data), np.std(train_data) + (10e-4)

    train_data = (train_data - train_mean) / std_mean
    val_data = (val_data - train_mean) / std_mean

    return train_data, val_data


def get_data_wave(data_x, num_sampled_points=200):
    length_multi_instance = num_sampled_points

    diff_data = np.zeros((len(data_x), length_multi_instance-1))

    for index in range(len(data_x)):

        if index % 10 == 0:
            print("COMPUTED FOR INDEX NUMBER: " + str(index))

        pred_before, pred_after, range_pred = get_one_dim_diff(data_x[index][0], data_x[index][1], num_sampled_points)
        feature = pred_before - pred_after
        feature = [feature[index] - feature[index+1] for index in range(len(feature)-1)]

        diff_data[index, :] = feature

    return diff_data


def make_grid(data_before, data_after, range_x, range_y, inverse=False):
    num_points_before = data_before.shape[0]
    num_points_after = data_after.shape[0]

    matrix_count = np.zeros((len(range_y) - 1, len(range_x) - 1))

    def get_subsets(data, range_x, range_y, i, j):
        subset = data[(data[:, 0] > range_x[i]) & (data[:, 0] < range_x[i + 1])]
        subset = subset[(subset[:, 1] > range_y[j]) & (subset[:, 1] < range_y[j + 1])]

        return subset

    for i in range(len(range_x) - 1):
        for j in range(len(range_y) - 1):
            subset_before = get_subsets(data_before, range_x, range_y, i, j)
            subset_after = get_subsets(data_after, range_x, range_y, i, j)

            matrix_count[j, i] = (subset_after.shape[0] / num_points_after) - (subset_before.shape[0] / num_points_before)

    if inverse:
        matrix_count = np.transpose(matrix_count)

    return matrix_count


def get_data_grid(data_x):

    range_x = np.arange(0, 6000, 1000)
    range_y = np.arange(0.92, 1.01, 0.01)
    data_list = []

    for index in range(len(data_x)):

        grid = make_grid(data_x[index][0], data_x[index][1], range_x, range_y)
        data_list.append(grid)

    return np.array(data_list)


def reshape_grid(grid):
    new_grid_array = np.zeros((grid.shape[0], 4, 5, 2))

    for index in range(grid.shape[0]):
        for depth in range(2):
            data = grid[index][depth * int(grid.shape[1] / 2):(depth + 1) * int(grid.shape[1] / 2)]
            new_grid_array[index][:, :, depth] = data

    return new_grid_array


def get_cut_lightcurve(lightcurve_before, lightcurve_after):
    max_min_time = max(np.min(lightcurve_before[:, 0]), np.min(lightcurve_after[:, 0]))
    min_max_time = min(np.max(lightcurve_before[:, 0]), np.max(lightcurve_after[:, 0]))

    lightcurve_before = lightcurve_before[(lightcurve_before[:, 0] >= max_min_time) & (lightcurve_before[:, 0] <= min_max_time)]
    lightcurve_after = lightcurve_after[(lightcurve_after[:, 0] >= max_min_time) & (lightcurve_after[:, 0] <= min_max_time)]

    return lightcurve_before, lightcurve_after


def interpolate_light_curve(light_curve, range_pred):
    data_x, data_y = np.expand_dims(light_curve[:, 0], axis=1), np.expand_dims(light_curve[:, 1], axis=1)
    range_pred = np.expand_dims(range_pred, axis=1)

    # spline = UnivariateSpline(data_x, data_y, s=0.001)
    # pred_new_range_spline = spline(range_pred)

    est = RandomForestRegressor(100)
    est.fit(data_x, data_y)
    pred_new_range = est.predict(range_pred)

    # pred_average = (pred_new_range_spline.flatten() + pred_new_range)/2

    # plt.plot()
    # plt.grid(True)
    # plt.scatter(data_x, data_y, s=5)
    # plt.scatter(range_pred, pred_new_range, s=5)
    # plt.scatter(range_pred, pred_new_range_spline, s=5)
    # plt.scatter(range_pred, pred_average, s=5)
    # plt.show()

    return pred_new_range


def get_one_dim_diff(data_before, data_after, num_intervals=100):
    data_before, data_after = get_cut_lightcurve(data_before, data_after)
    range_pred = np.linspace(np.min(data_before[:, 0]), np.max(data_before[:, 0]), num_intervals)

    pred_before = interpolate_light_curve(data_before, range_pred)
    pred_after = interpolate_light_curve(data_after, range_pred)

    return pred_before, pred_after, range_pred


def clip_values(predictions, dim):
    if dim == 0:
        predictions[predictions < 1] = 1.0
        predictions[predictions > 3] = 3.0
    elif dim == 2:
        predictions[predictions < 1.3] = 1.3
        predictions[predictions > 2.5] = 2.5
    else:
        predictions[:, 0][predictions[:, 0] < 1] = 1.0
        predictions[:, 0][predictions[:, 0] > 3] = 3.0

        predictions[:, 1][predictions[:, 1] < 1.3] = 1.3
        predictions[:, 1][predictions[:, 1] > 2.5] = 2.5

    return predictions
