#!/usr/bin/env python3.11

import open3d as o3d
from open3d.geometry import TriangleMesh
import numpy as np
import sys
from sklearn.neighbors import KDTree

# GLOSSARY OF VARIABLE NAME ABBREVIATIONS
# pcd = point cloud data
# pts = points
# pos = positions
# idx = indices
# norm = normalized
# grad = gradient
# vis = visualization
# tri = triangle
# vert = vertex
# slope = slope angle in radians
# cull = filtering/removal based on threshold
# mesh = triangular mesh
# bounds = bounding box
# geom = geometry

files = sys.argv[1:]

FILTER_ITERATIONS = 10
POINT_DISTANCE_THRESHOLD = 0.5
UPSIDE_DOWN = False
POINT_OVERLAY = False
GRADIENT_VISUALIZATION = True
SLOPE_CULLING_THRESHOLD = 1.5  # Minimum slope in radians to display points

PLOT_WIDTH = 1200
PLOT_HEIGHT = 800

for input_file_path in files:
    input_pcd = o3d.io.read_point_cloud(input_file_path)

    input_pcd.estimate_normals()

    oriented_bounding_box = input_pcd.get_oriented_bounding_box()
    input_pcd.rotate(oriented_bounding_box.R.T, center=oriented_bounding_box.center)

    if UPSIDE_DOWN:
        input_pcd.rotate(input_pcd.get_rotation_matrix_from_xyz(
            (np.pi, 0, 0)), center=input_pcd.get_center())

    reconstructed_mesh, poisson_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        input_pcd, depth=10)

    pcd_kdtree = o3d.geometry.KDTreeFlann(input_pcd)

    mesh_triangles = np.asarray(reconstructed_mesh.triangles)
    mesh_vertices = np.asarray(reconstructed_mesh.vertices)

    far_from_original_pcd_tri_mask = np.zeros(len(mesh_triangles), dtype=bool)
    for tri_idx, triangle_vert_idx in enumerate(mesh_triangles):
        vert_0_pos = mesh_vertices[triangle_vert_idx[0]]
        vert_1_pos = mesh_vertices[triangle_vert_idx[1]]
        vert_2_pos = mesh_vertices[triangle_vert_idx[2]]
        triangle_centroid_pos = (vert_0_pos + vert_1_pos + vert_2_pos) / 3.0
        [k, nearest_pcd_point_idx, _] = pcd_kdtree.search_knn_vector_3d(triangle_centroid_pos, 1)
        if k == 1:
            nearest_pcd_point_pos = np.asarray(input_pcd.points)[nearest_pcd_point_idx[0]]
            distance_to_original_pcd = np.linalg.norm(triangle_centroid_pos - nearest_pcd_point_pos)
            if distance_to_original_pcd > POINT_DISTANCE_THRESHOLD:
                far_from_original_pcd_tri_mask[tri_idx] = True

    reconstructed_mesh.remove_triangles_by_mask(far_from_original_pcd_tri_mask)
    reconstructed_mesh.remove_unreferenced_vertices()
    reconstructed_mesh.remove_degenerate_triangles()
    reconstructed_mesh.remove_duplicated_triangles()
    reconstructed_mesh.remove_duplicated_vertices()
    reconstructed_mesh.remove_non_manifold_edges()

    connected_tri_labels, connected_component_sizes, _ = reconstructed_mesh.cluster_connected_triangles()
    connected_tri_labels = np.asarray(connected_tri_labels)
    total_triangle_count = len(reconstructed_mesh.triangles)

    small_cluster_tri_mask = np.zeros(total_triangle_count, dtype=bool)

    for component_idx, component_size in enumerate(connected_component_sizes):
        if component_size < 3:
            small_cluster_tri_mask[connected_tri_labels == component_idx] = True

    if np.any(small_cluster_tri_mask):
        reconstructed_mesh.remove_triangles_by_mask(small_cluster_tri_mask)
        reconstructed_mesh.remove_unreferenced_vertices()

    reconstructed_mesh.filter_smooth_laplacian(number_of_iterations=FILTER_ITERATIONS)
    reconstructed_mesh.filter_smooth_taubin(number_of_iterations=FILTER_ITERATIONS)

    if POINT_OVERLAY:
        subsampled_display_pcd = input_pcd.random_down_sample(0.1)

    reconstructed_mesh.compute_vertex_normals()
    reconstructed_mesh.compute_triangle_normals()
    
    if GRADIENT_VISUALIZATION:
        mesh_triangle_normals = np.asarray(reconstructed_mesh.triangle_normals)
        
        vertical_axis = np.array([0, 0, 1])
        dot_products_with_vertical = np.abs(np.dot(mesh_triangle_normals, vertical_axis))
        clamped_dot_products = np.clip(dot_products_with_vertical, -1.0, 1.0)
        angles_from_vertical_axis = np.arccos(clamped_dot_products)
        triangle_slope_angles_rad = np.pi/2 - angles_from_vertical_axis
        
        min_slope_rad = triangle_slope_angles_rad.min()
        max_slope_rad = triangle_slope_angles_rad.max()
        if max_slope_rad > min_slope_rad:
            normalized_slope_values = (triangle_slope_angles_rad - min_slope_rad) / (max_slope_rad - min_slope_rad)
        else:
            normalized_slope_values = np.zeros_like(triangle_slope_angles_rad)
        
        slope_vis_pcd = reconstructed_mesh.sample_points_uniformly(number_of_points=50000)
        slope_vis_point_positions = np.asarray(slope_vis_pcd.points)
        
        current_mesh_triangles = np.asarray(reconstructed_mesh.triangles)
        triangle_centroid_positions = np.zeros((len(current_mesh_triangles), 3))
        for tri_idx, tri_vert_idx in enumerate(current_mesh_triangles):
            vert_0_pos = mesh_vertices[tri_vert_idx[0]]
            vert_1_pos = mesh_vertices[tri_vert_idx[1]]
            vert_2_pos = mesh_vertices[tri_vert_idx[2]]
            triangle_centroid_positions[tri_idx] = (vert_0_pos + vert_1_pos + vert_2_pos) / 3.0
        
        nearest_tri_finder = KDTree(triangle_centroid_positions)
        distances_to_nearest_tri, indices_of_nearest_tri = nearest_tri_finder.query(slope_vis_point_positions, k=1)
        slope_value_at_each_vis_point = normalized_slope_values[indices_of_nearest_tri]
        
        red_blue_grad_colors = np.zeros((len(slope_vis_point_positions), 3))
        red_blue_grad_colors[:, 0] = 1.0 - slope_value_at_each_vis_point.flatten()
        red_blue_grad_colors[:, 2] = slope_value_at_each_vis_point.flatten()
        red_blue_grad_colors = np.clip(red_blue_grad_colors, 0.0, 1.0)
        
        if SLOPE_CULLING_THRESHOLD > 0:
            actual_slope_at_each_point = triangle_slope_angles_rad[indices_of_nearest_tri]
            steep_enough_mask = actual_slope_at_each_point >= SLOPE_CULLING_THRESHOLD
            
            visible_point_positions = slope_vis_point_positions[steep_enough_mask.flatten()]
            visible_point_colors = red_blue_grad_colors[steep_enough_mask.flatten()]
            
            slope_vis_pcd = o3d.geometry.PointCloud()
            slope_vis_pcd.points = o3d.utility.Vector3dVector(visible_point_positions)
            slope_vis_pcd.colors = o3d.utility.Vector3dVector(visible_point_colors)
        else:
            slope_vis_pcd.colors = o3d.utility.Vector3dVector(red_blue_grad_colors)
    else:
        slope_vis_pcd = None

    mesh_bounds = reconstructed_mesh.get_axis_aligned_bounding_box()

    min_bound = mesh_bounds.min_bound
    max_bound = mesh_bounds.max_bound

    extents = max_bound - min_bound
    max_extent = np.max(extents)
    center = (min_bound + max_bound) * 0.5

    cube_min = center - max_extent * 0.5
    cube_max = center + max_extent * 0.5

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

    geometries = [reconstructed_mesh, bounding_box_mesh]
    
    if GRADIENT_VISUALIZATION and slope_vis_pcd is not None:
        geometries.append(slope_vis_pcd)
    
    if POINT_OVERLAY:
        geometries.append(subsampled_display_pcd)
    
    o3d.visualization.draw_plotly(geometries, width=PLOT_WIDTH, height=PLOT_HEIGHT)
