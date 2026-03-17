#!/usr/bin/env python3.11

import open3d as o3d
import numpy as np
import sys
import time
from sklearn.neighbors import KDTree

files = sys.argv[1:]

FILTER_ITERATIONS = 10
POINT_DISTANCE_THRESHOLD = 0.5
UPSIDE_DOWN = False
POINT_OVERLAY = False
GRADIENT_VISUALIZATION = True
SLOPE_CULLING_THRESHOLD = 1.5  # Minimum slope in radians to display points

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

    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

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
        display_cloud = point_cloud.random_down_sample(0.1)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    if GRADIENT_VISUALIZATION:
        triangle_normals = np.asarray(mesh.triangle_normals)
        
        z_axis = np.array([0, 0, 1])
        dot_products = np.abs(np.dot(triangle_normals, z_axis))
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles_from_vertical = np.arccos(dot_products)
        slopes = np.pi/2 - angles_from_vertical
        
        slope_min, slope_max = slopes.min(), slopes.max()
        if slope_max > slope_min:
            normalized_slopes = (slopes - slope_min) / (slope_max - slope_min)
        else:
            normalized_slopes = np.zeros_like(slopes)
        
        colored_mesh_points = mesh.sample_points_uniformly(number_of_points=50000)
        colored_points = np.asarray(colored_mesh_points.points)
        
        triangle_centroids = np.zeros((len(triangles), 3))
        for i, triangle in enumerate(np.asarray(mesh.triangles)):
            v0, v1, v2 = vertices[triangle]
            triangle_centroids[i] = (v0 + v1 + v2) / 3.0
        
        try:
            tree = KDTree(triangle_centroids)
            distances, indices = tree.query(colored_points, k=1)
            point_slopes = normalized_slopes[indices]
        except ImportError:
            point_slopes = np.zeros(len(colored_points))
            for i, point in enumerate(colored_points):
                distances = np.linalg.norm(triangle_centroids - point, axis=1)
                nearest_triangle_idx = np.argmin(distances)
                point_slopes[i] = normalized_slopes[nearest_triangle_idx]
        
        point_colors = np.zeros((len(colored_points), 3))
        point_colors[:, 0] = 1.0 - point_slopes.flatten()
        point_colors[:, 2] = point_slopes.flatten()
        point_colors = np.clip(point_colors, 0.0, 1.0)
        
        if SLOPE_CULLING_THRESHOLD > 0:
            actual_slopes = slopes[indices] if 'indices' in locals() else slopes
            
            slope_mask = actual_slopes >= SLOPE_CULLING_THRESHOLD
            
            filtered_points = colored_points[slope_mask.flatten()]
            filtered_colors = point_colors[slope_mask.flatten()]
            
            colored_mesh_points = o3d.geometry.PointCloud()
            colored_mesh_points.points = o3d.utility.Vector3dVector(filtered_points)
            colored_mesh_points.colors = o3d.utility.Vector3dVector(filtered_colors)
        else:
            colored_mesh_points.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        colored_mesh_points = None

    mesh_bounds = mesh.get_axis_aligned_bounding_box()

    min_bound = mesh_bounds.min_bound
    max_bound = mesh_bounds.max_bound

    extents = max_bound - min_bound
    max_extent = np.max(extents)
    center = (min_bound + max_bound) * 0.5

    cube_min = center - max_extent * 0.5
    cube_max = center + max_extent * 0.5

    from open3d.geometry import TriangleMesh
    bounding_points = np.array([
        [cube_min[0], cube_min[1], cube_min[2]],
        [cube_max[0], cube_min[1], cube_min[2]],
        [cube_min[0], cube_max[1], cube_min[2]],
        [cube_max[0], cube_max[1], cube_min[2]],
        [cube_min[0], cube_min[1], cube_max[2]],
        [cube_max[0], cube_min[1], cube_max[2]],
        [cube_min[0], cube_max[1], cube_max[2]],
        [cube_max[0], cube_max[1], cube_max[2]],
    ])

    bounding_box_mesh = TriangleMesh()
    bounding_box_mesh.vertices = o3d.utility.Vector3dVector(bounding_points)
    bounding_box_mesh.paint_uniform_color([1, 1, 1])

    geometries = [mesh, bounding_box_mesh]
    
    if GRADIENT_VISUALIZATION and colored_mesh_points is not None:
        geometries.append(colored_mesh_points)
    
    if POINT_OVERLAY:
        geometries.append(display_cloud)
    
    o3d.visualization.draw_plotly(geometries, width=PLOT_WIDTH, height=PLOT_HEIGHT)
