#!/usr/bin/env python3.11

import open3d as o3d
import numpy as np
import sys

files = sys.argv[1:]

FILTER_ITERATIONS = 1
UPSIDE_DOWN = True
POINT_OVERLAY = True

PLOT_WIDTH = 1200
PLOT_HEIGHT= 800

for file in files:
    point_cloud = o3d.io.read_point_cloud(file)
    
    point_cloud.estimate_normals()

    obb = point_cloud.get_oriented_bounding_box()
    point_cloud.rotate(obb.R.T, center=obb.center)

    if UPSIDE_DOWN:
        point_cloud.rotate(point_cloud.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=point_cloud.get_center())

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=10)

    mesh.filter_smooth_laplacian(number_of_iterations=FILTER_ITERATIONS)

    if POINT_OVERLAY:
        # subsampled points for visualisation
        display_cloud = point_cloud.random_down_sample(0.1)

    mesh.compute_vertex_normals()

    if POINT_OVERLAY:
        o3d.visualization.draw_plotly([mesh, display_cloud], width=PLOT_WIDTH , height=PLOT_HEIGHT)
    else:
        o3d.visualization.draw_plotly([mesh], width=PLOT_WIDTH , height=PLOT_HEIGHT)
        
