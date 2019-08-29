import os
import argparse
import scipy.spatial
import numpy as np


def read_frustum_data(file_path):
    with np.load(file_path) as data:
        points = data['points']

    return points


def get_unique_points(points):
    distances = scipy.spatial.distance.pdist(points[:, :3])
    close_points = distances < 1e-10

    close_points_square = scipy.spatial.distance.squareform(close_points)

    close_points_indices = np.nonzero(close_points_square)

    tosto = list({(x, y) if x < y else (y, x) for x, y in zip(close_points_indices[0], close_points_indices[1])})
    first = np.array([x[0] for x in tosto])
    second = np.array([x[1] for x in tosto])

    if len(first) == 0:
        return points

    duplicate_point_labels = (points[first, 3].astype(bool) | points[second, 3].astype(bool)).astype(np.float32)

    new_points_mask = np.ones(len(points), dtype=bool)
    new_points_mask[second] = False

    points[first, 3] = duplicate_point_labels

    return points[new_points_mask]


def sort_points(points):
    sorted_points = np.copy(points)
    sorted_points = sorted_points[sorted_points[:, 2].argsort()]
    sorted_points = sorted_points[sorted_points[:, 1].argsort(kind='mergesort')]
    sorted_points = sorted_points[sorted_points[:, 0].argsort(kind='mergesort')]

    return sorted_points


def get_arguments():
    parser = argparse.ArgumentParser(description='Script to merge different frustums of the same scene to a single file')

    parser.add_argument(
        'input', type=str,
        help='Path to directory containg points and labels per frustum in npz format'
    )

    parser.add_argument(
        'output', type=str,
        help='Path to save resulting full-scene points'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    input_dir = args.input
    if not input_dir or not os.path.isdir(input_dir):
        exit('Invalid input directory')

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    frustum_files = os.listdir(input_dir)
    frustum_files = [filename for filename in frustum_files if filename.endswith('.npz')]

    scene_ids = set([filename.split('_')[0] for filename in frustum_files])
    scene_ids = sorted(list(scene_ids))

    for scene_id in scene_ids:
        print(scene_id)
        scene_frustums = [filename for filename in frustum_files if filename.startswith(scene_id)]

        scene_points = list()
        for frustum in scene_frustums:
            frustum_file_path = os.path.join(input_dir, frustum)
            frustum_points = read_frustum_data(frustum_file_path)
            scene_points.append(frustum_points)

        scene_points = np.vstack(scene_points)
        unique_points = get_unique_points(scene_points)
        sorted_points = sort_points(unique_points)

        output_scene_path = os.path.join(output_dir, scene_id)
        np.savez(output_scene_path, points=sorted_points)
