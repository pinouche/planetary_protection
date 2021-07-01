import keras


def keras_model(seed, training_mode=False):
    num_filters = 64
    # input_shape = (4, 19, 1)
    input_shape = (200, 1)
    output_size = 1

    initializer = keras.initializers.glorot_normal(seed=seed)

    inputs = keras.layers.Input(input_shape)
    x = keras.layers.Conv1D(num_filters, 10, 2, activation='elu', kernel_initializer=initializer,
                            padding='same', input_shape=input_shape, trainable=True)(inputs)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x, training_mode)

    for layer in range(5):
        x = keras.layers.Conv1D(num_filters, 10, 2, activation='elu', kernel_initializer=initializer,
                                padding='same', input_shape=input_shape, trainable=True)(x)
        x = keras.layers.Conv1D(num_filters, 10, 2, activation='elu', kernel_initializer=initializer,
                                padding='same', input_shape=input_shape, trainable=True)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x, training_mode)

    print(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='elu', use_bias=True, kernel_initializer=initializer, trainable=True)(x)
    x = keras.layers.Dropout(0.5)(x, training_mode)

    # output layer
    output = keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                                kernel_initializer=initializer, trainable=True)(x)

    model = keras.Model(inputs=inputs, outputs=output)

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # optimizer = keras.optimizers.SGD(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model

