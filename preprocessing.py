import argparse
import os
import tqdm
import numpy as np


VELO_X_UNIT_VECTOR = np.array([1.0, 0.0])

# Deprecated


def scale_close(points: np.ndarray) -> np.ndarray:
    """Move points closer to the axes in the depth coordinate

    Arguments:
        points {np.ndarray} -- Points with labels, Nx4

    Returns:
        np.ndarray -- Points that have been moved closer, also with labels, Nx4
    """
    min_x = points[:, 0].min()

    return points - np.array([min_x, 0.0, 0.0, 0.0])


# Deprecated
def scale_to_box(points: np.ndarray) -> np.ndarray:
    """Scales points between [0, 1] in the depth coordinate and [-1, 1] in the other two coordinates
    Inferior to standard scaler

    Arguments:
        points {np.ndarray} -- Points with labels, Nx4

    Returns:
        np.ndarray -- Scaled points, Nx4
    """
    scale_factors = points.max(axis=0) - points.min(axis=0)

    yz_norm = 2 * (points - points.min(axis=0)) / scale_factors - 1
    x_norm = (points - points.min(axis=0)) / scale_factors

    return np.c_[x_norm[:, 0], yz_norm[:, 1:3], points[:, 3]]


def rotate_to_center(points: np.ndarray) -> np.ndarray:
    """Rotates the points in a frustum so that the frustum would line up with the depth axis
    This means that the mean value for points on the left-to-right axis is 0

    Arguments:
        points {np.ndarray} -- Points with labels, Nx4

    Returns:
        np.ndarray -- Rotated points with labels, Nx4
    """
    points_center_topdown = points.mean(axis=0)[:2]
    angle = np.arccos(VELO_X_UNIT_VECTOR.dot(points_center_topdown) / np.linalg.norm(points_center_topdown))

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.array([[cos_angle, -1.0 * sin_angle],
                                [sin_angle, cos_angle]])

    rotated_xy = points[:, :2].dot(rotation_matrix)
    return np.c_[rotated_xy, points[:, 2:]]


def scale_standard(points: np.ndarray) -> np.ndarray:
    """Scale points to follow the mean and standard deviation of the normal gaussian distribution

    Arguments:
        points {np.ndarray} -- Points with labels, Nx4

    Returns:
        np.ndarray -- Scaled points with labels, Nx4
    """
    scale_factors = points.std(axis=0)
    scaled_points = (points - points.mean(axis=0)) / scale_factors

    return np.c_[scaled_points[:, :3], points[:, 3]]


def get_arguments():
    parser = argparse.ArgumentParser(description='The normalization script for training and validation data used for training the model')

    parser.add_argument(
        'input', type=str,
        help='Path to directory containing points of each frustum as a separate file'
    )

    parser.add_argument(
        'output', type=str,
        help='Path to target directory where preprocessed frustum data should be saved'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    input_dir = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    frustum_files = sorted(os.listdir(input_dir))
    frustum_files = [frustum_file for frustum_file in frustum_files if frustum_file.endswith('.npz')]

    for frustum_file in tqdm.tqdm(frustum_files):
        frustum_file_path = os.path.join(input_dir, frustum_file)
        with np.load(frustum_file_path) as data:
            labelled_points = data['points']
            class_name = data['class_name']

        labelled_points = rotate_to_center(labelled_points)
        labelled_points = scale_standard(labelled_points)

        output_file_path = os.path.join(output_dir, frustum_file)
        np.savez(output_file_path, points=labelled_points, class_name=class_name)
