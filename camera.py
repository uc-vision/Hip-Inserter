import numpy as np
import cv2
import os
import json
import pyzed.sl as sl

from time import time


class Camera(object):
    def __init__(self):
        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
        self.init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 30

        self.fps = 30

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Could not connect to camera...")
            exit(1)

        self.runtime_parameters = sl.RuntimeParameters()

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()

        info = self.zed.get_camera_information()
        left_cam = info.camera_configuration.calibration_parameters.left_cam
        self.intrinsics = np.eye(3)
        self.intrinsics[0, 0] = left_cam.fx
        self.intrinsics[1, 1] = left_cam.fy
        self.intrinsics[0, 2] = left_cam.cx
        self.intrinsics[1, 2] = left_cam.cy

        self.baseline = self.zed.get_camera_information().camera_configuration.calibration_parameters.stereo_transform[0, 3]

        self.t0 = time()
        self.iter = 0

    def __call__(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)

            image_left_np = cv2.cvtColor(self.image_left.get_data()[:, :, :3], cv2.COLOR_RGB2BGR)
            image_right_np = cv2.cvtColor(self.image_right.get_data()[:, :, :3], cv2.COLOR_RGB2BGR)

            image_left_np = np.ascontiguousarray(image_left_np)
            image_right_np = np.ascontiguousarray(image_right_np)

            self.iter += 1

            return image_left_np, image_right_np
        
    def get_time(self):
        return time() - self.t0
        

class Recording(object):
    def __init__(self, dirpath, fps=30):

        left_filepath = os.path.join(dirpath, "left.mp4")
        self.left_cap = cv2.VideoCapture(left_filepath)

        right_filepath = os.path.join(dirpath, "right.mp4")
        self.right_cap = cv2.VideoCapture(right_filepath)

        self.fps = self.left_cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        json_filepath = os.path.join(dirpath, 'params.json')
        with open(json_filepath, 'r') as file:
            params = json.load(file)

        self.intrinsics = np.eye(3)
        self.intrinsics[0, 0] = params['fx']
        self.intrinsics[1, 1] = params['fy']
        self.intrinsics[0, 2] = params['cx']
        self.intrinsics[1, 2] = params['cy']

        self.baseline = params['baseline']

        self.t0 = time()
        self.iter = 0

    def __len__(self):
        return int(self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __call__(self):

        if self.iter >= self.num_frames:
            return None, None

        left_ret, left_frame = self.left_cap.read()
        right_ret, right_frame = self.right_cap.read()

        if not (left_ret and right_ret):
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            left_ret, left_frame = self.left_cap.read()
            right_ret, right_frame = self.right_cap.read()

        image_left_np = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
        image_right_np = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)

        image_left_np = np.ascontiguousarray(image_left_np)
        image_right_np = np.ascontiguousarray(image_right_np)

        self.iter += 1

        return image_left_np, image_right_np
    
    def get_time(self):
        return self.t0 + self.iter / self.fps
