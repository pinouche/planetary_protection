import keras
import numpy as np

from utils import clip_values


def keras_model(seed, training_mode=False):
    num_filters = 16
    input_shape = (200, 1)
    output_size = 1

    initializer = keras.initializers.glorot_normal(seed=seed)

    inputs = keras.layers.Input(input_shape)
    x = keras.layers.Conv1D(num_filters, 20, 2, activation='elu', kernel_initializer=initializer,
                            padding='same', input_shape=input_shape, trainable=True)(inputs)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x, training_mode)

    for layer in range(5):
        x = keras.layers.Conv1D(num_filters, 20, 2, activation='elu', kernel_initializer=initializer,
                                padding='same', input_shape=input_shape, trainable=True)(x)

        # x = keras.layers.MaxPooling1D(pool_size=2, padding="valid")(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x, training_mode)

    print(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='elu', use_bias=True, kernel_initializer=initializer, trainable=True)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x, training_mode)

    outputs = keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                                 kernel_initializer=initializer, trainable=True)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # optimizer = keras.optimizers.SGD(lr=0.001)

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model


def cnn_training_schemes(x_train_inner, x_val_inner, y_train_inner, y_val_inner, dim_of_interest, num_eval, stochastic_averaging=False,
                         averaging_at_inference_time=True):

    x_train_inner, x_val_inner = x_train_inner / np.abs(np.max(x_train_inner)), x_val_inner / np.abs(np.max(x_train_inner))
    model = keras_model(num_eval, False)

    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    def scheduler(epoch, lr):
        if epoch >= 0:
            lr = 0.0005
        return lr

    scheduler_callback = keras.callbacks.LearningRateScheduler(scheduler)

    model.fit(x_train_inner, y_train_inner[:, dim_of_interest], batch_size=16, epochs=200,
                verbose=2, validation_data=(x_val_inner, y_val_inner[:, dim_of_interest]), callbacks=[early_stop_callback])

    if stochastic_averaging:

        weight_list = []
        for _ in range(10):
            model.fit(x_train_inner, y_train_inner[:, dim_of_interest], batch_size=16, epochs=1,
                      verbose=2, validation_data=(x_val_inner, y_val_inner[:, dim_of_interest]), callbacks=[scheduler_callback])

            weights = model.get_weights()
            weight_list.append(weights)

        weights = np.mean(np.array(weight_list), axis=0)
        model = keras_model(num_eval, False)
        model.set_weights(weights)

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

    keras.backend.clear_session()

    return predictions_train, predictions_val
