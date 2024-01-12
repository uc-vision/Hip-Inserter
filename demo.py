import hydra
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

from visualise import DrawVector, DrawDepth, DrawValues

from camera import Recording
from inference import Inference
from transforms import Project, roll_pitch_angles


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def run(cfg):

    # model_filepath = "./models/sleap_projects/models/7-points-3230831_174033.single_instance.n=300"

    if cfg.camera is True:
        pass
    else:
        camera = Recording(cfg.camera)
    inference = Inference(cfg.model_filepath)
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
    depth_moving_avg = MovingAvg(0.5)

    depth_estimate = DepthEstimate(2)

    draw_vector = DrawVector(project, 500)
    draw_depth = DrawDepth()
    draw_values = DrawValues()

    for i in tqdm(range(len(camera))):
        # Images and time from camera
        image_left, image_right = camera()
        t = camera.get_time()

        # 2D points inference, and processing
        points_left, points_right = inference.stereo_inference(image_left, image_right)
        points_left = buffer_left(points_left, t)
        points_right = buffer_right(points_right, t)
        points_left, points_right = horozontal_check(points_left, points_right)
        points_left, points_right = principle_axis_var_check.both(points_left, points_right)

        # 3D points projection and processing
        points_3d = project.project(points_left, points_right)
        points_3d = distance_check(points_3d)
        points_3d = geometry_check(points_3d)

        # Angles processing
        mean, vector = points_2_vector(points_3d)
        vector_current = points_2_vector.get_current()
        roll, pitch = roll_pitch_angles(vector)
        roll = roll_moving_avg(roll, t)
        pitch = pitch_moving_avg(pitch, t)

        # Depth processing
        depth, depth_pixel = depth_estimate(points_3d, points_left)
        depth_current = depth_estimate.get_current()
        depth = depth_moving_avg(depth, t)

        # Visualisation processing
        image_left_out = np.copy(image_left)
        image_left_out = draw_vector(image_left_out, mean, vector, vector_current)
        image_left_out = draw_depth(image_left_out, depth_pixel, depth_current)
        image_values = draw_values(roll, pitch, depth, vector_current, depth_current)
        image_out = np.concatenate([image_left_out, image_values], axis=1)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)

        # Show visualisation
        cv2.imshow("image", image_out)
        cv2.waitKey(1)


if __name__ == "__main__":
    run()


