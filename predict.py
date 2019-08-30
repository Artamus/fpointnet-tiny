import os
import argparse
import preprocessing
import time
import numpy as np
import tensorflow as tf
from fpointnet_tiny_functional import get_compiled_model
from scipy import stats


def read_raw_data(data_path, allowed_class, sample_limit=None):
    data_filenames = sorted(os.listdir(data_path))
    data_filenames = [filename for filename in data_filenames if filename.endswith('.npz')]

    frustums_data = list()
    kept_frustums = list()

    num_samples = 0

    for filename in data_filenames:
        file_path = os.path.join(data_path, filename)
        with np.load(file_path) as data:
            class_name = data['class_name']
            point_data = data['points']

        if class_name != allowed_class:
            continue

        frustums_data.append(point_data)
        kept_frustums.append(filename)

        num_samples += 1

        if sample_limit and num_samples >= sample_limit:
            break

    return frustums_data, kept_frustums


def sample_points(labelled_points, num_points, sample_at_least_once=False):
    scene_points = np.array(labelled_points)

    if sample_at_least_once:
        if len(scene_points) > num_points:
            mask = np.random.choice(len(scene_points), num_points, replace=False)
        elif len(scene_points) == num_points:
            mask = np.arange(len(scene_points))
            np.random.shuffle(mask)
        else:
            mask = np.zeros(shape=num_points, dtype=np.int32)
            mask[:len(scene_points)] = np.arange(len(scene_points), dtype=np.int32)
            mask[len(scene_points):] = np.random.choice(len(scene_points), num_points - len(scene_points), replace=True)
            np.random.shuffle(mask)
    else:
        mask = np.random.choice(len(scene_points), num_points, replace=True)

    sampled_labelled_points = scene_points[mask]

    return sampled_labelled_points, mask


def structure_data(scenes_labelled_points, num_points):
    points = np.zeros(shape=(len(scenes_labelled_points), num_points, 3))
    labels = np.zeros(shape=(len(scenes_labelled_points), num_points))
    masks = np.zeros(shape=(len(scenes_labelled_points), num_points))

    for i, labelled_points in enumerate(scenes_labelled_points):
        sampled_labelled_points, mask = sample_points(labelled_points, num_points, True)
        points[i] = sampled_labelled_points[:, :3]
        labels[i] = sampled_labelled_points[:, 3]
        masks[i] = mask

    points = np.expand_dims(points, axis=2)
    return points, labels, masks


def all_samples_softmax(x):
    x_exp = np.exp(x)
    probabilities = x_exp / x_exp.sum(axis=2)[:, :, None]
    return np.argmax(probabilities, axis=2)


def match_predictions_points(frustums, predicted_labels, masks):

    predicted_frustums = list()

    for points, predictions, mask in zip(frustums, predicted_labels, masks):
        points = np.array(points)

        for point_index in range(len(points)):
            points_matching_original = np.where(mask == point_index)[0]

            if len(points_matching_original) == 0:
                mode_label = 0
            else:
                mode_label = stats.mode(predictions[points_matching_original]).mode[0]

            points[point_index, 3] = float(mode_label)

        predicted_frustums.append(points)

    return predicted_frustums


def save_predictions(output_dir, filenames, frustum_data):

    for filename, data in zip(filenames, frustum_data):
        output_file_path = os.path.join(output_dir, filename)
        np.savez(output_file_path, points=data)


def calculate_accuracy(predictions, values):
    return (predictions == values).mean()


def calculate_true_accuracy(predictions, values):
    assert len(predictions) == len(values), 'Predictions and ground truth don\'t have the same length'

    counts = np.zeros(shape=(len(predictions), 2))

    for index in range(len(predictions)):
        counts[index] = [(predictions[index][:, 3] == values[index][:, 3]).sum(), len(predictions[index])]

    total_counts = counts.sum(axis=0)
    return 1.0 * total_counts[0] / total_counts[1]


def get_arguments():
    parser = argparse.ArgumentParser(description='The program to predict from validation data.')

    parser.add_argument(
        'input', type=str,
        help='Path to directory containing data to perform predictions on (XYZ points with label per point saved in the .npz format)'
    )

    parser.add_argument(
        'output', type=str,
        help='Directory to save output to, will be created if it does not exist'
    )

    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to the model file or directory containing models (in case of >1 models they will be sorted alphabetically and last will be used)'
    )

    parser.add_argument(
        '-np', '--num_points', type=int, default=512,
        help='Number of points to sample from each frustum'
    )

    parser.add_argument(
        '--class_name', default='person',
        choices=['person', 'car'],
        help='Class to use from the KITTI dataset'
    )

    parser.add_argument(
        '--eval', action='store_true', default=False,
        help='Perform evaluation of the predictions'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    input_dir = args.input
    if not input_dir or not os.path.isdir(input_dir):
        exit('Invalid input directory')

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    model_path = args.model
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        files = sorted([filename for filename in files if filename.endswith('.h5') or filename.endswith('.hdf5')])
        model_path = os.path.join(model_path, files[-1])

    num_points = args.num_points
    allowed_class = args.class_name

    frustums_data, filenames = read_raw_data(input_dir, allowed_class)

    processed_frustums_data = list()
    for frustum in frustums_data:
        processed_frustum = preprocessing.rotate_to_center(frustum)
        processed_frustum = preprocessing.scale_standard(processed_frustum)
        processed_frustums_data.append(processed_frustum)

    data_x, data_y, masks = structure_data(processed_frustums_data, num_points)

    model = get_compiled_model(num_points, 3e-4)  # learning rate is just for reusing the model code
    model.load_weights(model_path)

    start_time = time.perf_counter()
    prediction_logits = model.predict(data_x)
    end_time = time.perf_counter()

    print(f'Inference took {end_time - start_time} s')

    predictions = all_samples_softmax(prediction_logits)

    frustums_with_predicted_labels = match_predictions_points(frustums_data, predictions, masks)
    save_predictions(output_dir, filenames, frustums_with_predicted_labels)

    if not args.eval:
        exit()

    accuracy = calculate_accuracy(predictions, data_y)
    print(f'Accuracy on structured points is {accuracy:.3f}')

    true_accuracy = calculate_true_accuracy(frustums_with_predicted_labels, frustums_data)
    print(f'Accuracy on raw points is {true_accuracy:.3f}')
