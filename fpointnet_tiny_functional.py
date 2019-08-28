import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, Input
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization


def get_model(num_points):
    """Create the Keras model

    Arguments:
        num_points {int} -- Number of points to sample from each frustum to keep a consistent input

    Returns:
        Keras Model -- The uncompiled Keras model
    """
    inputs = Input(shape=(num_points, 1, 3))
    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(inputs)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    point_features = BatchNormalization(axis=3)(x)

    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(point_features)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)

    global_features = tf.reduce_max(x, axis=1)
    global_features = tf.expand_dims(global_features, axis=2)

    global_features_expanded = tf.tile(global_features, [1, num_points, 1, 1])
    combined_features = tf.concat([point_features, global_features_expanded], axis=3)

    x = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), activation='relu')(combined_features)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Dropout(0.5)(x)
    x = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(x)
    outputs = tf.squeeze(x, axis=2)

    return Model(inputs=inputs, outputs=outputs, name='FP-func')


def get_compiled_model(num_points, learning_rate):
    """Create the Keras model with loss and optimizer and compile it

    Arguments:
        num_points {int} -- Number of points to sample from each frustum to keep a consistent input
        learning_rate {float} -- The learning rate to use to train the model

    Returns:
        Keras Model -- The compiled Keras model
    """
    model = get_model(num_points)

    # The commented loss is the original loss used in older versions of TensorFlow
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=, logits=))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    mock_x_train = tf.constant([
        [[0.2, 3.4, 1.9], [2.5, 1.5, 9.6]],
        [[1.2, 1.3, 1.4], [2.5, 1.3, 0.5]],
        [[1.2, 1.3, 1.4], [2.5, 1.3, 0.5]],
        [[1.2, 1.3, 1.4], [2.5, 1.3, 0.5]],
        [[1.2, 1.3, 1.4], [2.5, 1.3, 0.5]],
    ], dtype=tf.float32)

    mock_y_train = tf.constant([
        [[0], [1]],
        [[1], [0]],
        [[1], [0]],
        [[1], [0]],
        [[1], [0]],
    ], dtype=np.int32)

    mock_one_hot = tf.constant([
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=tf.float32)

    print(mock_x_train.shape)

    model = get_compiled_model(2, 3e-4)
    print(model.summary())

    md = tf.expand_dims(mock_x_train, axis=2)
    model.fit(md, mock_y_train, epochs=5)
