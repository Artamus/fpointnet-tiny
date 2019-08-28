import os
import datetime
import argparse
import numpy as np
import tensorflow as tf
from fpointnet_tiny_functional import get_compiled_model


FLIPPING_TENSOR = tf.constant([1.0, -1.0, 1.0])


def read_raw_data(data_path, allowed_class, sample_limit=None):
    data_filenames = sorted(os.listdir(data_path))
    data_filenames = [filename for filename in data_filenames if filename.endswith('.npz')]

    data_x = list()
    data_y = list()
    num_samples = 0

    for filename in data_filenames:
        file_path = os.path.join(data_path, filename)
        with np.load(file_path) as data:
            class_name = data['class_name']
            point_data = data['points']

        if class_name != allowed_class:
            continue

        data_x.append(point_data[:, :3].tolist())
        data_y.append(point_data[:, 3].tolist())

        num_samples += 1

        if sample_limit and num_samples >= sample_limit:
            break

    return data_x, data_y


@tf.function
def sample_data(points, labels, num_points):

    big_points = list()
    big_labels = list()

    for ind in range(points.shape[0]):
        scene_points = points[ind]
        scene_labels = labels[ind]
        scene_size = tf.size(scene_points)
        maxval = tf.math.floordiv(scene_size, 3)
        mask = tf.random.uniform((num_points,), maxval=maxval, dtype=tf.int32)

        new_points = tf.expand_dims(tf.gather(scene_points, mask), axis=1)
        new_labels = tf.gather(scene_labels, mask)

        big_points.append(new_points)
        big_labels.append(new_labels)

    return tf.stack(big_points), tf.stack(big_labels)


@tf.function
def flip(points, labels):
    if tf.random.uniform(shape=()) >= 0.5:
        return points * FLIPPING_TENSOR, labels

    return points, labels


def get_arguments():
    parser = argparse.ArgumentParser(description='The main training program for this fpointnet-tiny architecture.')

    parser.add_argument(
        'train', type=str,
        help='Path to directory containing training data (XYZ points with label per point saved in the .npz format)'
    )

    parser.add_argument(
        'val', type=str,
        help='Path to directory containing validation data (XYZ points with label per point saved in the .npz format)'
    )

    parser.add_argument(
        '-np', '--num_points', type=int, default=512,
        help='Number of points to sample from each frustum'
    )

    parser.add_argument(
        '-e', '--epochs', type=int, default=50,
        help='Number of epochs to train the model for'
    )

    parser.add_argument(
        '-b', '--batch', type=int, default=32,
        help='Number of samples per batch'
    )

    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=3e-4,
        help='Learning rate to use for the model'
    )

    parser.add_argument(
        '--class_name', default='person',
        choices=['person', 'car'],
        help='Class to use from the KITTI dataset'
    )

    parser.add_argument(
        '--run_id', type=str,
        help='Specify an ID to use for this run, datetime if left empty'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    train_data_path = args.train
    if not train_data_path or not os.path.isdir(train_data_path):
        exit('Invalid train path')

    val_data_path = args.val
    if not val_data_path or not os.path.isdir(val_data_path):
        exit('Invalid validation path')

    num_points = args.num_points
    num_epochs = args.epochs
    batch_size = args.batch
    learning_rate = args.learning_rate
    allowed_class = args.class_name
    run_id = args.run_id

    train_x, train_y = read_raw_data(train_data_path, allowed_class, 500)
    print(f'Raw training data has {len(train_x)} samples')

    val_x, val_y = read_raw_data(val_data_path, allowed_class, 100)
    print(f'Raw validation data has {len(val_x)} samples')

    train_x = tf.ragged.constant(train_x, ragged_rank=1)
    train_y = tf.ragged.constant(train_y, ragged_rank=1)
    print(f'Sanity check for ragged tensors, x shape: {train_x.shape}, y shape: {train_y.shape}')

    val_x = tf.ragged.constant(val_x, ragged_rank=1)
    val_y = tf.ragged.constant(val_y, ragged_rank=1)

    steps_per_epoch = np.ceil(train_x.shape[0] / batch_size).astype(np.int32)
    print(f'Sanity check steps per epoch: {steps_per_epoch}')

    print('#### Assembling Dataset object ####')
    # TODO: Figure out how many to prefetch

    sampling_lambda = lambda x, y: sample_data(x, y, num_points)

    train_data = tf.data.Dataset.from_tensors((train_x, train_y)) \
        .map(sampling_lambda) \
        .unbatch() \
        .map(flip) \
        .batch(batch_size) \
        .repeat(num_epochs) \
        .prefetch(4)
    
    val_data = tf.data.Dataset.from_tensors((val_x, val_y)) \
        .map(sampling_lambda) \
        .unbatch() \
        .batch(batch_size) \
        .prefetch(4)

    train_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if not run_id:
        run_id = f'{allowed_class}-{train_time}'

    log_dir = os.path.join('logs', run_id)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_path = os.path.join('models', run_id, 'model-{epoch:03d}.h5')
    os.makedirs(os.path.join('models', run_id), exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                     monitor='val_loss',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=0)

    # TODO: Try different strategies for LR reducing
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=5, min_lr=1e-5, min_delta=0.001, verbose=1)

    callbacks = [
        tensorboard_callback,
        cp_callback,
        reduce_lr_callback
    ]

    print('#### Training model ####')
    model = get_compiled_model(num_points, learning_rate)
    model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=val_data, callbacks=callbacks)
