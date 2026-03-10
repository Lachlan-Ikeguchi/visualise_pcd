#! /usr/bin/env python3.11

import open3d as o3d
import sys

files = sys.argv[1:]

for file in files:
    point_cloud = o3d.io.read_point_cloud(file)
    
    point_cloud.estimate_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

    o3d.visualization.draw_plotly([mesh])   
