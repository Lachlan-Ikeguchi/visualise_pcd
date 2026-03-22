#!/usr/bin/env python3.11

import open3d as o3d
import numpy as np
import sys

# GLOSSARY OF VARIABLE NAME ABBREVIATIONS
# pcd = point cloud data
# idx = index
# o3d = Open3D library
# np = NumPy library

# Style guide:
# - Black formatter
# - Clear human-readable code execution steps
# - Verbose variable names with shortnames defined in the glossary
# - Do not use comments and write self-documenting code


def flip_point_cloud_upside_down(point_cloud):
    point_cloud.rotate(
        point_cloud.get_rotation_matrix_from_xyz((np.pi, 0, 0)),
        center=point_cloud.get_center(),
    )
    return point_cloud


def process_file(input_file_path, output_file_path):
    input_pcd = o3d.io.read_point_cloud(input_file_path)
    input_pcd.estimate_normals()
    oriented_bounding_box = input_pcd.get_oriented_bounding_box()
    input_pcd.rotate(oriented_bounding_box.R.T, center=oriented_bounding_box.center)

    flipped_pcd = flip_point_cloud_upside_down(input_pcd)

    o3d.io.write_point_cloud(output_file_path, flipped_pcd)


def main():
    args = sys.argv[1:]

    if "-o" not in args:
        print(
            "Usage: python point_cloud_flipper.py <input_file1> <input_file2> ... -o <output_file1> <output_file2> ..."
        )
        sys.exit(1)

    split_idx = args.index("-o")
    input_file_paths = args[:split_idx]
    output_file_paths = args[split_idx + 1 :]

    if len(input_file_paths) != len(output_file_paths):
        print("Error: Number of input files must match number of output files")
        sys.exit(1)

    for input_file_path, output_file_path in zip(input_file_paths, output_file_paths):
        process_file(input_file_path, output_file_path)


if __name__ == "__main__":
    main()
