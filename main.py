import numpy as np
import cv2
import sys
import os
import math as m
# import pyzed.sl as sl
import time

from tqdm import tqdm

from sleap.nn.inference import load_model

from camera import Recording
from inference import Inference
from visualizers import DrawPoints, DrawLine, DrawAngles
from transforms import Project
from transforms import points_to_vector, roll_pitch_angles


if __name__ == "__main__":

    # model_filepath = "./data/sleap_checkpoints/test_data_reencode/models/no_arg_230403_104958.single_instance.n=141"
    # model_filepath = "./models/room_model"
    model_filepath = "./models/sleap_projects/models/7-points-3230831_174033.single_instance.n=300"
    # model_filepath = "./models/sleap_projects/models/je1230511_151929.single_instance.n=11"

    # camera = Camera()
    camera = Recording("./data/collection/20231231_191639")
    # camera = Recording("./data/collection/20231231_191742")
    # camera = Recording("./data/collection/20231231_191825")
    inference = Inference(model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    # fourcc = cv2.VideoWriter_fourcc(*'h264')
    # cap = cv2.VideoWriter("./data/collection/20231231_191639/raw_output.mp4", fourcc, 30.0, (1920, 540))
    # cap = cv2.VideoWriter("./data/collection/20231231_191742/raw_output.mp4", fourcc, 30.0, (1920, 540))
    # cap = cv2.VideoWriter("./data/collection/20231231_191825/raw_output.mp4", fourcc, 30.0, (1920, 540))

    dp = DrawPoints(scale=0.75, multi=True)
    # dl = DrawLine(scale=0.5, multi=True)

    # visualizers = []
    # visualizers.append(DrawPoints(scale=0.75, multi=True))
    # visualizers.append(DrawLine(scale=0.75, multi=True))
    # visualizers.append(DrawAngles(multi=True))

    for i in tqdm(range(len(camera))):
        image_left, image_right = camera()
        points_left, points_right = inference.stereo_inference(image_left, image_right)
        # print(points_left.shape)
        points_3d = project.project(points_left, points_right)
        # points_left = points_left[:5, :]
        # points_right = points_right[:5, :]
        vector = points_to_vector(points_3d)

        # print(image_left.shape)

        if i % 1 == 0:

            image_points = dp.draw_stereo(image_left, image_right, points_left, points_right)
            image_points = cv2.cvtColor(image_points, cv2.COLOR_BGR2RGB)

            if dp.scale != 1:
                x_resize = int(image_points.shape[1] * dp.scale)
                y_resize = int(image_points.shape[0] * dp.scale)
                image_points = cv2.resize(image_points, (x_resize, y_resize))

            # cap.write(np.ascontiguousarray(image_points))

            cv2.imshow("image left points", image_points)
            cv2.waitKey(1)

            # for visualizer in visualizers:
            #     if isinstance(visualizer, DrawPoints):
            #         visualizer.send(image_left, image_right, points_left, points_right)
            #     if isinstance(visualizer, DrawLine):
            #         visualizer.send(image_left, points_left)
            #     if isinstance(visualizer, DrawAngles):
            #         visualizer.send(vector)

        # time.sleep(0.1)
            
    # cap.release()
    
    camera.left_cap.release()
    camera.right_cap.release()
    cv2.destroyAllWindows()
            
