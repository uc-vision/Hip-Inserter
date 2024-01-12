import numpy as np
import math as m
import json
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from processing import HorozontalCheck
from processing import PrincipleAxisVarCheck
from processing import MovingAvg
from processing import Buffer
from processing import DistanceCheck
from processing import GeometryCheck
from processing import Points2Vector
from processing import Buffer, Buffers
from processing import DepthEstimate

from visualise import DrawVector, DrawDepth

from camera import Recording
from inference import Inference
from transforms import Project, roll_pitch_angles


def closest_point_vector(q, v1, v2):
    ''' calculate interpolated 't' value along vector between points v1 and v2 '''
    n = (v1[0] - q[:, 0])*(v2[0] - v1[0]) + (v1[1] - q[:, 1])*(v2[1] - v1[1]) + (v1[2] - q[:, 2])*(v2[2] - v1[2])
    d = (v2[0] - v1[0])**2 + (v2[1] - v1[1])**2 + (v2[2] - v1[2])**2
    t = -n / d
    return t


if __name__ == "__main__":

    model_filepath = "./models/sleap_projects/models/7-points-3230831_174033.single_instance.n=300"

    # camera = Recording("./data/collection/20240107_183743")
    camera = Recording("./data/collection/20240107_183830")

    inference = Inference(model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    buffer_left = Buffers(0.2, 2, 5)
    buffer_right = Buffers(0.2, 2, 5)

    horozontal_check = HorozontalCheck(2)
    principle_axis_var_check = PrincipleAxisVarCheck(0.001, 3)

    distance_check = DistanceCheck(0.1)
    geometry_check = GeometryCheck(0.02)

    points_2_vector = Points2Vector()

    roll_moving_avg = MovingAvg(0.2)
    pitch_moving_avg = MovingAvg(0.5)

    draw_vector = DrawVector(project, 500)

    depth_estimate = DepthEstimate()

    draw_depth = DrawDepth(project)

    num_points = 5

    dists = np.zeros([len(camera), num_points, num_points])

    for i in tqdm(range(len(camera))):
        image_left, image_right = camera()

        t = camera.get_time()

        points_left, points_right = inference.stereo_inference(image_left, image_right)

        points_left = buffer_left(points_left, t)
        points_right = buffer_right(points_right, t)

        points_left, points_right = horozontal_check(points_left, points_right)
        points_left, points_right = principle_axis_var_check.both(points_left, points_right)

        points_3d = project.project(points_left, points_right)

        points_3d = distance_check(points_3d)
        points_3d = geometry_check(points_3d)

        mean, vector = points_2_vector(points_3d)
        current = points_2_vector.get_current()

        vector_norm = vector / np.linalg.norm(vector)

        v1 = mean
        v2 = mean + vector_norm

        t = closest_point_vector(points_3d, v1, v2)

        dists[i] = t[:, None] - t[None, :]

    dists_avg = np.nanmedian(dists, axis=0)

    print(dists_avg)