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
from transforms import Project, roll_pitch_angles


def view_points(camera, points):

    dp = DrawPoints(scale=0.75, multi=True)

    for i in tqdm(range(len(camera))):
        image_left, image_right = camera()

        points_left = np.array(points[i]["left"], dtype=float)
        points_right = np.array(points[i]["right"], dtype=float)

        points_left[points_left < 0] = m.nan
        points_right[points_right < 0] = m.nan

        image_points = dp.draw_stereo(image_left, image_right, points_left, points_right)
        image_points = cv2.cvtColor(image_points, cv2.COLOR_BGR2RGB)

        if dp.scale != 1:
            x_resize = int(image_points.shape[1] * dp.scale)
            y_resize = int(image_points.shape[0] * dp.scale)
            image_points = cv2.resize(image_points, (x_resize, y_resize))

        cv2.imshow("test", image_points)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


def npafy(points):
    for i in range(len(points)):
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



def closest_point_vector(q, v1, v2):
    ''' calculate interpolated 't' value along vector between points v1 and v2 '''
    n = (v1[0] - q[:, 0])*(v2[0] - v1[0]) + (v1[1] - q[:, 1])*(v2[1] - v1[1]) + (v1[2] - q[:, 2])*(v2[2] - v1[2])
    d = (v2[0] - v1[0])**2 + (v2[1] - v1[1])**2 + (v2[2] - v1[2])**2
    t = -n / d
    return t



if __name__ == "__main__":

    run_dir = "20231231_191639"
    # run_dir = "20231231_191742"
    # run_dir = "20231231_191825"

    with open(f"./analysis/outputs/{run_dir}/points.json", 'r') as fp:
        points = json.load(fp)

    # for i in range(len(points)):

    #     points_left = np.array(points[i]["left"], dtype=float)
    #     points_right = np.array(points[i]["right"], dtype=float)

    #     points_left[points_left < 0] = m.nan
    #     points_right[points_right < 0] = m.nan

    #     points[i]["left"] = points_left
    #     points[i]["right"] = points_right

    # points_new = deepcopy(points)
    
    npafy(points)

    camera = Recording(f"./data/collection/{run_dir}")
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    colors = []
    for i in range(5):
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        colors.append(color)

    horozontal_check(points)

    # points_dist = np.zeros((len(points), points[0]["left"].shape[0], points[0]["left"].shape[0]))
    for i in range(len(points)):

        image_left, image_right = camera()

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

        image_point = image_left

        for j in range(point_left_sub.shape[0]):
            image_point = cv2.circle(
                        img=image_left,
                        center=(int(point_left_sub[j, 0]), int(point_left_sub[j, 1])),
                        radius=0,
                        color=[0, 255, 255],
                        thickness=8)

        if point_3d_sub.shape[0] >= 3:

            points_dist = np.linalg.norm(point_3d_sub[None, :, :] - point_3d_sub[:, None, :], axis=-1)
            # print(np.amax(points_dist))

            if np.amax(points_dist) > 0.12:
                # print(point_3d_sub.shape)
                line = cv2.fitLine(point_3d_sub, cv2.DIST_L2, 0, 0.01, 0.01)
                # print(line)
                # exit()

                pca = decomposition.PCA(3)

                pca.fit(point_3d_sub)

                # print(pca.components_)
                # print(pca.explained_variance_)
                # print(pca.mean_)

                mean = pca.mean_[None, :]
                # print(mean.shape)

                line_set1 = pca.mean_[None, :] + pca.components_[None, 0] * np.linspace(-1, 1, 500)[:, None]
                # line_set2 = line[None, 3:, 0] + line[None, :3, 0] * np.linspace(-1, 1, 500)[:, None]

                p1 = project.unproject(line_set1)
                # p2 = project.unproject(line_set2)

                for j in range(500):
                    image_point = cv2.circle(
                                img=image_left,
                                center=(int(p1[j, 0]), int(p1[j, 1])),
                                radius=0,
                                color=[255, 255, 0],
                                thickness=3)
                    
                # for j in range(500):
                #     image_point = cv2.circle(
                #                 img=image_left,
                #                 center=(int(p2[j, 0]), int(p2[j, 1])),
                #                 radius=0,
                #                 color=[0, 255, 0],
                #                 thickness=3)
                # print(point_left_sub.shape)
                    
                # for j in range(point_left_sub.shape[0]):
                #     image_point = cv2.circle(
                #                 img=image_left,
                #                 center=(int(point_left_sub[j, 0]), int(point_left_sub[j, 1])),
                #                 radius=0,
                #                 color=[0, 0, 255],
                #                 thickness=8)
            
        image_point = cv2.cvtColor(image_point, cv2.COLOR_BGR2RGB)
            
        cv2.imshow("test", image_point)
        cv2.waitKey(100)

    cv2.destroyAllWindows()

        # plt.imshow(image_point)
        # plt.show()

        # print(p)

        # exit()

        # A = np.concatenate([point_3d[:, :2], np.ones([point_3d.shape[0], 1])], axis=-1)
        # # A = np.concatenate([point_3d[:, :1], np.ones([point_3d.shape[0], 1])], axis=-1)
        # # print(A)
        # z = point_3d[:, 2]

        # coef, res, _, _ = np.linalg.lstsq(A, z, rcond=None)
        # # coef = np.array([m1, m2, c])
        # # print(m1, m2, c)

        # v1 = c
        # v2 = coef[0] + coef[1] + c

        # print(res)
        # exit()

        # point_dist = np.linalg.norm(point_3d[None, :, :] - point_3d[:, None, :], axis=-1)
        # points_dist[i] = point_dist

    # print(points_dist.shape)

    # for i in range(len(points)):

    #     for j in range(len(points[i]["left"])):
    #         if m.isnan(points[i]["left"][j][0]) or m.isnan(points[i]["right"][j][0]):
    #             continue

    #         vertical_delta = abs(points[i]["left"][j][1] - points[i]["right"][j][1])
            
    #         if vertical_delta > 1:
    #             points[i]["left"][j][0] = m.nan
    #             points[i]["left"][j][1] = m.nan
    #             points[i]["right"][j][0] = m.nan
    #             points[i]["right"][j][1] = m.nan

    #     points[i]["left"][points[i]["left"] < 0] = m.nan
    #     points[i]["right"][points[i]["right"] < 0] = m.nan

    #     point_3d = project.project(points[i]["left"], points[i]["right"])
        
    #     points_dist = np.linalg.norm(point_3d[None, :, :] - point_3d[:, None, :], axis=-1)
    #     print(points_dist)
    #     print()

    # exit()

    # view_points(camera, points)

    # camera.left_cap.release()
    # camera.right_cap.release()