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

FILES = sys.argv[1:]

FILTER_ITERATIONS = 5
POINT_DISTANCE_THRESHOLD = 0.5 # meters
UPSIDE_DOWN = False
POINT_OVERLAY = False
GRADIENT_VISUALIZATION = True
SLOPE_CULLING_THRESHOLD = 1.5 # radians

PLOT_WIDTH = 1200
PLOT_HEIGHT = 800


def create_triangle_centroids(tri_vertices, all_vertices):
    return (all_vertices[tri_vertices[:, 0]] + 
            all_vertices[tri_vertices[:, 1]] + 
            all_vertices[tri_vertices[:, 2]]) / 3.0


def load_and_preprocess_point_cloud(file_path):
    input_pcd = o3d.io.read_point_cloud(file_path)
    input_pcd.estimate_normals()
    oriented_bounding_box = input_pcd.get_oriented_bounding_box()
    input_pcd.rotate(oriented_bounding_box.R.T, center=oriented_bounding_box.center)
    
    if UPSIDE_DOWN:
        input_pcd.rotate(input_pcd.get_rotation_matrix_from_xyz(
            (np.pi, 0, 0)), center=input_pcd.get_center())
    
    return input_pcd


def create_mesh_from_point_cloud(point_cloud):
    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=8)


def filter_triangles_by_distance(mesh, point_cloud):
    pcd_kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    mesh_triangles = np.asarray(mesh.triangles)
    mesh_vertices = np.asarray(mesh.vertices)
    
    far_from_original_pcd_tri_mask = np.zeros(len(mesh_triangles), dtype=bool)
    triangle_centroids = create_triangle_centroids(mesh_triangles, mesh_vertices)
    
    batch_size = 1000
    for batch_start in range(0, len(triangle_centroids), batch_size):
        batch_end = min(batch_start + batch_size, len(triangle_centroids))
        batch_centroids = triangle_centroids[batch_start:batch_end]
        
        for local_idx, centroid in enumerate(batch_centroids):
            [k, nearest_pcd_point_idx, _] = pcd_kdtree.search_knn_vector_3d(centroid, 1)
            if k == 1:
                nearest_pcd_point_pos = np.asarray(point_cloud.points)[nearest_pcd_point_idx[0]]
                distance_to_original_pcd = np.linalg.norm(centroid - nearest_pcd_point_pos)
                if distance_to_original_pcd > POINT_DISTANCE_THRESHOLD:
                    global_idx = batch_start + local_idx
                    far_from_original_pcd_tri_mask[global_idx] = True
    
    mesh.remove_triangles_by_mask(far_from_original_pcd_tri_mask)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    return mesh


def remove_small_clusters(mesh):
    connected_tri_labels, connected_component_sizes, _ = mesh.cluster_connected_triangles()
    connected_tri_labels = np.asarray(connected_tri_labels)
    
    small_cluster_tri_mask = np.zeros(len(mesh.triangles), dtype=bool)
    
    for component_idx, component_size in enumerate(connected_component_sizes):
        if component_size < 3:
            small_cluster_tri_mask[connected_tri_labels == component_idx] = True
    
    if np.any(small_cluster_tri_mask):
        mesh.remove_triangles_by_mask(small_cluster_tri_mask)
        mesh.remove_unreferenced_vertices()
    
    return mesh


def smooth_mesh(mesh):
    mesh.filter_smooth_laplacian(number_of_iterations=FILTER_ITERATIONS)
    mesh.filter_smooth_taubin(number_of_iterations=FILTER_ITERATIONS)
    return mesh


def create_gradient_visualization(mesh):
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    mesh_triangle_normals = np.asarray(mesh.triangle_normals)
    
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
    
    slope_vis_pcd = mesh.sample_points_uniformly(number_of_points=50000)
    slope_vis_point_positions = np.asarray(slope_vis_pcd.points)
    
    current_mesh_triangles = np.asarray(mesh.triangles)
    mesh_vertices = np.asarray(mesh.vertices)
    triangle_centroid_positions = create_triangle_centroids(current_mesh_triangles, mesh_vertices)
    
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
    
    return slope_vis_pcd


def create_bounding_box(mesh):
    mesh_bounds = mesh.get_axis_aligned_bounding_box()
    
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
    
    return bounding_box_mesh


def create_point_cloud_overlay(point_cloud):
    return point_cloud.random_down_sample(0.1)


def visualize_mesh(mesh, bounding_box, slope_vis_pcd=None, point_cloud_overlay=None):
    geometries = [mesh, bounding_box]
    
    if GRADIENT_VISUALIZATION and slope_vis_pcd is not None:
        geometries.append(slope_vis_pcd)
    
    if POINT_OVERLAY and point_cloud_overlay is not None:
        geometries.append(point_cloud_overlay)
    
    o3d.visualization.draw_plotly(geometries, width=PLOT_WIDTH, height=PLOT_HEIGHT)


def process_file(file_path):
    point_cloud = load_and_preprocess_point_cloud(file_path)
    mesh, _ = create_mesh_from_point_cloud(point_cloud)
    mesh = filter_triangles_by_distance(mesh, point_cloud)
    mesh = remove_small_clusters(mesh)
    mesh = smooth_mesh(mesh)
    
    slope_vis_pcd = None
    if GRADIENT_VISUALIZATION:
        slope_vis_pcd = create_gradient_visualization(mesh)
    
    bounding_box = create_bounding_box(mesh)
    
    point_cloud_overlay = None
    if POINT_OVERLAY:
        point_cloud_overlay = create_point_cloud_overlay(point_cloud)
    
    visualize_mesh(mesh, bounding_box, slope_vis_pcd, point_cloud_overlay)


def main():
    for file_path in FILES:
        process_file(file_path)


if __name__ == "__main__":
    main()
