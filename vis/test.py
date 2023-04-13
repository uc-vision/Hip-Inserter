import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1280, height=720)

    ctr = vis.get_view_control()
    
    camera_front = np.array([[0.0029999820000323999, -110.002999995500002026, 0.99999100002700003]], dtype=np.float64).T
    camera_lookat = np.array([[0.13334449567213114, -0.090414432181677593, -0.48072464491026173]], dtype=np.float64).T
    camera_up = np.array([[8.999973000032397e-06, 0.999995500003375, 0.0029999820000324012]], dtype=np.float64).T
    zoom = 0.94000000000000017
    ctr.set_front(camera_front)
    ctr.set_lookat(camera_lookat)
    ctr.set_up(camera_up)

    cylinder_radius = 0.005
    cylinder_height = 0.5
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_height)

    vis.add_geometry(cylinder)

    vis.run()