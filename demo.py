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
from processing import DistanceCheck
from processing import GeometryCheck
from processing import Points2Vector
from processing import Buffer, Buffers
from processing import DepthEstimate

from visualise import DrawVector, DrawDepth, DrawValues

from camera import Camera, Recording
from inference import Inference
from transforms import Project, roll_pitch_angles


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def run(cfg):

    if cfg.camera is True:
        camera = Camera()
    else:
        camera = Recording(cfg.camera)
    inference = Inference(cfg.model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    buffer_left = Buffers(
        cfg.buffers.time_delta_max,
        cfg.buffers.pixel_delta_max,
        cfg.buffers.num_buffers)
    buffer_right = Buffers(
        cfg.buffers.time_delta_max,
        cfg.buffers.pixel_delta_max,
        cfg.buffers.num_buffers)

    horozontal_check = HorozontalCheck(
        cfg.horozontal_check.pixel_delta_max)
    principle_axis_var_check = PrincipleAxisVarCheck(
        cfg.principle_axis_var_check.var_max, 
        cfg.principle_axis_var_check.points_min)

    geometry_check = GeometryCheck(
        cfg.geometry_check.distance_delta_max,
        cfg.geometry_check.points_min)
    distance_check = DistanceCheck(cfg.distance_check.distance_min)

    points_2_vector = Points2Vector()

    roll_moving_avg = MovingAvg(cfg.moving_average.roll)
    pitch_moving_avg = MovingAvg(cfg.moving_average.pitch)
    depth_moving_avg = MovingAvg(cfg.moving_average.depth)

    depth_estimate = DepthEstimate(cfg.depth_estimate.points_idx)

    draw_vector = DrawVector(project, cfg.draw_vector.n_points)
    draw_depth = DrawDepth()
    draw_values = DrawValues()

    if cfg.record.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap = cv2.VideoWriter(cfg.record.output, fourcc, cfg.record.fps, (1280+480, 720))

    t = camera.get_time()

    while True:
        # Images and time from camera
        image_left, image_right = camera()
        if image_left is None or image_right is None:
            return
        t = camera.get_time()

        # 2D points inference, and processing
        points_left, points_right = inference.stereo_inference(image_left, image_right)
        points_left = buffer_left(points_left, t)
        points_right = buffer_right(points_right, t)
        points_left, points_right = horozontal_check(points_left, points_right)
        points_left, points_right = principle_axis_var_check.both(points_left, points_right)

        # 3D points projection and processing
        points_3d = project.project(points_left, points_right)
        points_3d = geometry_check(points_3d)
        points_3d = distance_check(points_3d)

        # Angles processing
        mean, vector = points_2_vector(points_3d)
        vector_current = points_2_vector.get_current()
        roll, pitch = roll_pitch_angles(vector)
        if vector_current:
            roll = roll_moving_avg(roll, t)
            pitch = pitch_moving_avg(pitch, t)
        else:
            roll = roll_moving_avg(np.nan, t)
            pitch = pitch_moving_avg(np.nan, t)

        # Depth processing
        depth, depth_pixel = depth_estimate(points_3d, points_left)
        depth_current = depth_estimate.get_current()
        if depth_current:
            depth = depth_moving_avg(depth, t)
        else:
            depth = depth_moving_avg(np.nan, t)

        # Visualisation processing
        image_left_out = np.copy(image_left)
        image_left_out = draw_vector(image_left_out, mean, vector, vector_current)
        image_left_out = draw_depth(image_left_out, depth_pixel, depth_current)
        image_values = draw_values(roll, pitch, depth, vector_current, depth_current)
        image_out = np.concatenate([image_left_out, image_values], axis=1)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)

        # Visualisation Output
        if cfg.show_output:
            cv2.imshow("image", image_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if cfg.record.output is not None:
            cap.write(np.ascontiguousarray(image_out))

    if cfg.show_output:
        cv2.destroyAllWindows()
    if cfg.record.output is not None:
        cap.release()

    



if __name__ == "__main__":
    run()


