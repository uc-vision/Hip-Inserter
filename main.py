import numpy as np
import cv2
import sys
import os
import math as m
import pyzed.sl as sl
import time

from tqdm import tqdm

from sleap.nn.inference import load_model

from camera import Camera, Recording
from inference import Inference
from visualizers import DrawPoints, DrawLine, DrawAngles
from transforms import Project
from transforms import points_to_vector, roll_pitch_angles


if __name__ == "__main__":

    # model_filepath = "./data/sleap_checkpoints/test_data_reencode/models/no_arg_230403_104958.single_instance.n=141"
    model_filepath = "./models/room"

    # camera = Camera()
    camera = Recording("./data/side2_vid/recording")
    inference = Inference(model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    visualizers = []
    visualizers.append(DrawPoints(scale=0.75, multi=True))
    # visualizers.append(DrawLine(scale=0.75, multi=True))
    visualizers.append(DrawAngles(multi=True))

    for i in tqdm(range(10000)):
        image_left, image_right = camera()
        points_left, points_right = inference.stereo_inference(image_left, image_right)
        points_3d = project.project(points_left, points_right)
        vector = points_to_vector(points_3d)

        if i % 1 == 0:
            for visualizer in visualizers:
                if isinstance(visualizer, DrawPoints):
                    visualizer.send(image_left, image_right, points_left, points_right)
                if isinstance(visualizer, DrawLine):
                    visualizer.send(image_left, points_left)
                if isinstance(visualizer, DrawAngles):
                    visualizer.send(vector)

        time.sleep(0.08)
