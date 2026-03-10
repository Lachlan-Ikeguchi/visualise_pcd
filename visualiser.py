#!/usr/bin/env python3.11

import open3d as o3d
import numpy as np
import sys

files = sys.argv[1:]

FILTER_ITERATIONS = 1
UPSIDE_DOWN = True
POINT_OVERLAY = True

for file in files:
    point_cloud = o3d.io.read_point_cloud(file)
    
    point_cloud.estimate_normals()

    obb = point_cloud.get_oriented_bounding_box()
    point_cloud.rotate(obb.R.T, center=obb.center)

    if UPSIDE_DOWN:
        point_cloud.rotate(point_cloud.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=point_cloud.get_center())

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=10)

    mesh.filter_smooth_laplacian(number_of_iterations=FILTER_ITERATIONS)

    # remove poorly-defined huge faces (often from open regions/unbounded areas)
    # compute triangle areas manually since Open3D lacks a direct helper
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if tris.size > 0:
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        thresh = areas.mean() + 3 * areas.std()
        mask = areas < thresh
        mesh.remove_triangles_by_mask(~mask)
        mesh.remove_unreferenced_vertices()

    if POINT_OVERLAY:
        # subsampled points for visualisation
        display_cloud = point_cloud.random_down_sample(0.1)

    mesh.compute_vertex_normals()

    if POINT_OVERLAY:
        o3d.visualization.draw_plotly([mesh, display_cloud])
    else:
        o3d.visualization.draw_plotly([mesh])
        