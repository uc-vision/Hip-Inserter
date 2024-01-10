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

from visualise import DrawVector

from camera import Recording
from inference import Inference
from transforms import Project, roll_pitch_angles


if __name__ == "__main__":

    model_filepath = "./models/sleap_projects/models/7-points-3230831_174033.single_instance.n=300"

    camera = Recording("./data/collection/20240107_183743")

    inference = Inference(model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    horozontal_check = HorozontalCheck(2)
    principle_axis_var_check = PrincipleAxisVarCheck(0.001, 3)

    distance_check = DistanceCheck(0.1)
    geometry_check = GeometryCheck(0.02)

    points_2_vector = Points2Vector()

    roll_moving_avg = MovingAvg(0.2)
    pitch_moving_avg = MovingAvg(0.5)

    draw_vector = DrawVector(project, 500)

    for i in tqdm(range(len(camera))):
        image_left, image_right = camera()

        points_left, points_right = inference.stereo_inference(image_left, image_right)

        points_left, points_right = horozontal_check(points_left, points_right)
        points_left, points_right = principle_axis_var_check.both(points_left, points_right)

        points_3d = project.project(points_left, points_right)

        points_3d = distance_check(points_3d)
        points_3d = geometry_check(points_3d)

        mean, vector = points_2_vector(points_3d)
        current = points_2_vector.get_current()

        image_out = draw_vector(image_left, mean, vector, current)

        roll, pitch = roll_pitch_angles(vector)

        roll = roll_moving_avg(roll, camera.get_time())
        pitch = pitch_moving_avg(pitch, camera.get_time())

        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        cv2.imshow("image", image_out)
        cv2.waitKey(1)




