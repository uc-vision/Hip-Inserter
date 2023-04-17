import pyzed.sl as sl
import cv2
import math
import numpy as np
import sys
import os
import json

from tqdm import tqdm
from datetime import datetime


def record(dirpath_root, num_seconds):
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode

    date_str = datetime.today().strftime('%Y%m%d_%H%M%S')
    dirpath_record = os.path.join(dirpath_root, date_str)
    os.makedirs(dirpath_record, exist_ok=True)
    filepath_left = os.path.join(dirpath_record, "left.mp4")
    filepath_right = os.path.join(dirpath_record, "right.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'h264')
    cap_left = cv2.VideoWriter(filepath_left, fourcc, 30.0, (1280, 720))
    cap_right = cv2.VideoWriter(filepath_right, fourcc, 30.0, (1280, 720))

    # Get stereo camera params
    info = zed.get_camera_information()
    left_cam = info.calibration_parameters.left_cam

    # Save params in json file
    stereo_cam = {}
    stereo_cam['fx'] = left_cam.fx
    stereo_cam['fy'] = left_cam.fy
    stereo_cam['cx'] = left_cam.cx
    stereo_cam['cy'] = left_cam.cy
    stereo_cam['baseline'] = info.calibration_parameters.T[0].item()
    params_path = os.path.join(dirpath_record, "params.json")
    with open(params_path, 'w') as file:
        json.dump(stereo_cam, file)


    image_left = sl.Mat()
    image_right = sl.Mat()

    num_frames = num_seconds * 30

    for i in tqdm(range(num_frames)):
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            images_left_np = np.ascontiguousarray(image_left.get_data()[:, :, :3])
            images_right_np = np.ascontiguousarray(image_right.get_data()[:, :, :3])

            cap_left.write(images_left_np)
            cap_right.write(images_right_np)

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    dirpath_root = "./data/collection"
    num_seconds = 10

    record(dirpath_root, num_seconds)