import numpy as np
import math as m
import json
import cv2
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm
from sklearn import datasets, decomposition
import random

from visualizers import DrawPoints, DrawLine, DrawAngles
from camera import Recording
from transforms import Project


def npafi(points):
    for i in range(len(points)):
        for j in range(len(points)):
            points_left = np.array(points[i]["left"], dtype=float)
            points_right = np.array(points[i]["right"], dtype=float)

            points_left[points_left < 0] = m.nan
            points_right[points_right < 0] = m.nan

            points[i]["left"] = points_left
            points[i]["right"] = points_right

def horozontal_check(points):
    for i in range(len(points)):
        for j in range(len(points[i]["left"])):
            if m.isnan(points[i]["left"][j][0]) or m.isnan(points[i]["right"][j][0]):
                continue

            vertical_delta = abs(points[i]["left"][j][1] - points[i]["right"][j][1])
            
            if vertical_delta > 1:
                points[i]["left"][j][0] = m.nan
                points[i]["left"][j][1] = m.nan
                points[i]["right"][j][0] = m.nan
                points[i]["right"][j][1] = m.nan

if __name__ == "__main__":

    run_dir = "20231231_191639"
    # run_dir = "20231231_191742"
    # run_dir = "20231231_191825"

    with open(f"./analysis/outputs/{run_dir}/points.json", 'r') as fp:
        points = json.load(fp)

    npafi(points)

    camera = Recording(f"./data/collection/{run_dir}")
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    horozontal_check(points)

    points_dist = np.zeros([len(camera), 5, 5])
    points_dist[...] = np.nan

    initial = np.array([
        [0, 0.03, 0.07, 0.16, 0.28],
        [0.03, 0, 0.04, 0.13, 0.25],
        [0.07, 0.04, 0, 0.09, 0.21],
        [0.16, 0.13, 0.09, 0, 0.12],
        [0.28, 0.25, 0.21, 0.12, 0],
        ])

    for i in range(len(points)):

        point_left = points[i]["left"]
        point_right = points[i]["right"]
        point_3d = project.project(point_left, point_right)


        point_left = point_left[:5, :]
        point_right = point_right[:5, :]
        point_3d = point_3d[:5, :]

        # if np.isnan(point_3d).any():
        #     continue

        point_left_sub = np.zeros((0, 2))
        point_right_sub = np.zeros((0, 2))

        point_3d_sub = np.zeros((0, 3))

        for j in range(point_3d.shape[0]):
            if np.isnan(point_3d[j]).any():
                continue
            point_3d_sub = np.concatenate([point_3d_sub, point_3d[None, j]], axis=0)

            point_left_sub = np.concatenate([point_left_sub, point_left[None, j]], axis=0)
            point_right_sub = np.concatenate([point_right_sub, point_right[None, j]], axis=0)

        if point_3d_sub.shape[0] >= 5:

            point_dist = np.linalg.norm(point_3d[None, :, :] - point_3d[:, None, :], axis=-1)

            point_dist[np.abs(point_dist - initial) > 0.02] = np.nan

            points_dist[i] = point_dist

            # for j in range(point_3d.shape[0]):
            #     if np.isnan(point_3d[j]).any():
            #         continue

            #     for k in range(point_3d.shape[0]):
            #         if np.isnan(point_3d[k]).any():
            #             continue

    print(np.nanmedian(points_dist, axis=0))

    # print(points_dist)