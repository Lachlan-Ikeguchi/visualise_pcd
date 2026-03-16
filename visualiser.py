#!/usr/bin/env python3.11

import open3d as o3d
import numpy as np
import sys

files = sys.argv[1:]

FILTER_ITERATIONS = 10

POINT_DISTANCE_THRESHOLD = 0.5

UPSIDE_DOWN = False
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

    # Build KDTree on point cloud for distance queries
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Compute centroids and check distance to nearest point
    far_triangle_mask = np.zeros(len(triangles), dtype=bool)
    for i, triangle in enumerate(triangles):
        v0, v1, v2 = vertices[triangle]
        centroid = (v0 + v1 + v2) / 3.0
        [k, idx, _] = pcd_tree.search_knn_vector_3d(centroid, 1)
        if k == 1:
            nearest_point = np.asarray(point_cloud.points)[idx[0]]
            distance = np.linalg.norm(centroid - nearest_point)
            if distance > POINT_DISTANCE_THRESHOLD:
                far_triangle_mask[i] = True

    mesh.remove_triangles_by_mask(far_triangle_mask)
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

    extents = max_bound - min_bound
    max_extent = np.max(extents)
    center = (min_bound + max_bound) * 0.5

    # Keep the plot scaled equally in all dimensions by using a cubic bounding box.
    cube_min = center - max_extent * 0.5
    cube_max = center + max_extent * 0.5

    from open3d.geometry import TriangleMesh
    bounding_points = np.array([
        [cube_min[0], cube_min[1], cube_min[2]],  # Bottom
        [cube_max[0], cube_min[1], cube_min[2]],
        [cube_min[0], cube_max[1], cube_min[2]],
        [cube_max[0], cube_max[1], cube_min[2]],
        [cube_min[0], cube_min[1], cube_max[2]],  # Top
        [cube_max[0], cube_min[1], cube_max[2]],
        [cube_min[0], cube_max[1], cube_max[2]],
        [cube_max[0], cube_max[1], cube_max[2]],
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
