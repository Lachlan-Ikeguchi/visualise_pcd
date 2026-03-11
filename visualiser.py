#!/usr/bin/env python3.11

import open3d as o3d
import numpy as np
import sys

files = sys.argv[1:]

FILTER_ITERATIONS = 10

FILTER_LENGTH_THRESHOLD = 0.2

UPSIDE_DOWN = True
POINT_OVERLAY = True

PLOT_WIDTH = 1200
PLOT_HEIGHT = 800

for file in files:
    point_cloud = o3d.io.read_point_cloud(file)

    point_cloud.estimate_normals()

    obb = point_cloud.get_oriented_bounding_box()
    point_cloud.rotate(obb.R.T, center=obb.center)

    if UPSIDE_DOWN:
        point_cloud.rotate(point_cloud.get_rotation_matrix_from_xyz(
            (np.pi, 0, 0)), center=point_cloud.get_center())

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=10)

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    edge_lengths = []
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle]
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2)
        ]
        edge_lengths.append(max(edges))

    large_triangle_mask = np.array(edge_lengths) > FILTER_LENGTH_THRESHOLD
    mesh.remove_triangles_by_mask(large_triangle_mask)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    triangle_labels, cluster_sizes, _ = mesh.cluster_connected_triangles()
    triangle_labels = np.asarray(triangle_labels)
    num_triangles = len(mesh.triangles)

    triangles_to_remove_mask = np.zeros(num_triangles, dtype=bool)

    for i, size in enumerate(cluster_sizes):
        if size < 3:
            triangles_to_remove_mask[triangle_labels == i] = True

    if np.any(triangles_to_remove_mask):
        mesh.remove_triangles_by_mask(triangles_to_remove_mask)
        mesh.remove_unreferenced_vertices()

    mesh.filter_smooth_laplacian(number_of_iterations=FILTER_ITERATIONS)
    mesh.filter_smooth_taubin(number_of_iterations=FILTER_ITERATIONS)

    if POINT_OVERLAY:
        # subsampled points for visualisation
        display_cloud = point_cloud.random_down_sample(0.1)

    mesh.compute_vertex_normals()

    mesh_bounds = mesh.get_axis_aligned_bounding_box()

    min_bound = mesh_bounds.min_bound
    max_bound = mesh_bounds.max_bound

    z_range = max_bound[2] - min_bound[2]

    # expand z range to make it flatter
    z_expansion = z_range * 2.0

    from open3d.geometry import TriangleMesh
    bounding_points = np.array([
        [min_bound[0], min_bound[1], min_bound[2] - z_expansion],  # Bottom
        [max_bound[0], min_bound[1], min_bound[2] - z_expansion],
        [min_bound[0], max_bound[1], min_bound[2] - z_expansion],
        [max_bound[0], max_bound[1], min_bound[2] - z_expansion],
        [min_bound[0], min_bound[1], max_bound[2] + z_expansion],  # Top
        [max_bound[0], min_bound[1], max_bound[2] + z_expansion],
        [min_bound[0], max_bound[1], max_bound[2] + z_expansion],
        [max_bound[0], max_bound[1], max_bound[2] + z_expansion],
    ])

    bounding_box_mesh = TriangleMesh()
    bounding_box_mesh.vertices = o3d.utility.Vector3dVector(bounding_points)
    bounding_box_mesh.paint_uniform_color([1, 1, 1])  # invisible

    if POINT_OVERLAY:
        o3d.visualization.draw_plotly(
            [mesh, display_cloud, bounding_box_mesh],
            width=PLOT_WIDTH, height=PLOT_HEIGHT)
    else:
        o3d.visualization.draw_plotly(
            [mesh, bounding_box_mesh],
            width=PLOT_WIDTH, height=PLOT_HEIGHT)
