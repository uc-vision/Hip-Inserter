# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import cv2
import h5py
import os
import sys
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pyzed.sl as sl
import platform
import random
import threading
import time


def load_keypoints(filepath):
    with h5py.File(filepath, 'r') as f:
        tracks_matrix = f['tracks'][:]
    return tracks_matrix


def load_images(dirpath):
    images = []
    for i in range(len(os.listdir(dirpath))):
        filename = f"{i}.png"
        filepath = os.path.join(dirpath, filename)
        # images.append(cv2.imread(filepath)[..., ::-1])
        images.append(cv2.imread(filepath))
    return images


def get_stereo_params():
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    info = zed.get_camera_information()

    focal = info.calibration_parameters.left_cam.fx
    baseline = info.calibration_parameters.T[0]
    width = info.calibration_parameters.left_cam.image_size.width

    left_cam = info.calibration_parameters.left_cam
    intrinsics = np.eye(3)
    intrinsics[0, 0] = left_cam.fx
    intrinsics[1, 1] = left_cam.fy
    intrinsics[0, 2] = left_cam.cx
    intrinsics[1, 2] = left_cam.cy

    return focal, baseline, width, intrinsics


def disparity_to_depth(focal, baseline, disparity):
    depth = baseline * focal / disparity
    return depth


def draw_image(image_left, image_right, keypoint_left, keypoint_right):
    image = np.concatenate([image_left, image_right], axis=1)

    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255]]

    for j in [0, 4]:
        try:
            x_l = int(keypoint_left[0, 0, j])
            y_l = int(keypoint_left[0, 1, j])
            image = cv2.circle(img=image, center=(x_l, y_l), radius=0, color=colors[j], thickness=8)
        except ValueError:
            pass
        # if not (m.isnan(x_l) or m.isnan(y_l)):
        try:
            x_r = int(keypoint_right[0, 0, j]) + 1280
            y_r = int(keypoint_right[0, 1, j])
            image = cv2.circle(img=image, center=(x_r, y_r), radius=0, color=colors[j], thickness=8)

            image = cv2.line(image, (x_l, y_l), (x_r, y_r), color=colors[j])
        except ValueError:
            pass

    return image


def stereo_inference():
    focal, baseline, width, K = get_stereo_params()
    keypoints_left = load_keypoints("./data/outputs/side2/2/train_labels.v001.003_test_left.analysis.h5")
    keypoints_right = load_keypoints("./data/outputs/side2/2/train_labels.v001.002_test_right.analysis.h5")

    poses = []

    vectors = []

    print(keypoints_left.shape[3])

    for i in range(keypoints_left.shape[3]):
        left_1 = keypoints_left[0, :, 0, i]
        left_5 = keypoints_left[0, :, 4, i]

        right_1 = keypoints_right[0, :, 0, i]
        right_5 = keypoints_right[0, :, 4, i]

        depth_1 = disparity_to_depth(focal, baseline, left_1[0]-right_1[0])
        depth_5 = disparity_to_depth(focal, baseline, left_5[0]-right_5[0])

        p1 = np.concatenate([left_1, np.array((1,))], axis=0)
        p5 = np.concatenate([left_5, np.array((1,))], axis=0)

        P1 = np.linalg.inv(K) @ (p1 * depth_1)
        P5 = np.linalg.inv(K) @ (p5 * depth_5)

        # Move the cylinder's center to the midpoint between the points
        midpoint = (P5 + P1) / 2
        translation = np.eye(4)
        translation[:3, 3] = midpoint

        distance = np.linalg.norm(P1 - P5)
        direction_vector = (P5 - P1) / distance
        axis_alignment = np.cross(np.array([0, 0, 1]), direction_vector)
        rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), direction_vector))
        rotation_matrix = np.eye(4)
        axis_alignment = axis_alignment / np.linalg.norm(axis_alignment)
        K_ = np.array([
            [0, -axis_alignment[2], axis_alignment[1]],
            [axis_alignment[2], 0, -axis_alignment[0]],
            [-axis_alignment[1], axis_alignment[0], 0]
        ])
        rotation_matrix[:3, :3] = np.eye(3) + np.sin(rotation_angle) * K_ + (1 - np.cos(rotation_angle)) * np.dot(K_, K_)

        # pose = rotation_matrix @ translation
        pose = translation @ rotation_matrix
        poses.append(np.copy(pose))

        vectors.append(direction_vector)

        # P1[0] = -P1[0]
        # P1[1] = -P1[1]
        # P1[2] = -P1[2]
        # P5[0] = -P5[0]
        # P5[1] = -P5[1]
        # P5[2] = -P5[2]

    return K, poses, vectors


def get_graphs(vectors):
    graphs = []
    for i in range(len(vectors)):
        fig, ax = plt.subplots()
        ax.bar(['X', 'Y', 'Z'], vectors[i], color=['red', 'green', 'blue'])
        ax.set_ylim(-1, 1)

        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())[..., :3][..., ::-1]
        graphs.append(X)
        plt.close()
    return graphs


class SpheresApp:
    MENU_SPHERE = 1
    MENU_RANDOM = 2
    MENU_QUIT = 3

    def __init__(self):
        self._id = 0
        # self.window = gui.Application.instance.create_window(
        #     "Spatial Rendering", 1280, 720)
        self.window = gui.Application.instance.create_window(
            "Spatial Rendering", 960, 540)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.scene.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        
        self.scene.scene.show_axes(True)
        # self.scene.scene.show_skybox(True)
        # gp = rendering.Scene.GroundPlane(0)
        # self.scene.scene.show_ground_plane(True, gp)

        # plane = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # mat = rendering.MaterialRecord()
        # self.scene.scene.add_geometry('plane', plane, mat)
        
        self.intrinsics, self.poses, self.vectors = stereo_inference()

        self.intrinsics[0, 0] = self.intrinsics[0, 0] * 0.75
        self.intrinsics[1, 1] = self.intrinsics[1, 1] * 0.75

        self.intrinsics[0, 2] = self.intrinsics[0, 2] * 0.75
        self.intrinsics[1, 2] = self.intrinsics[1, 2] * 0.75

        self.images_left = load_images("./data/side2/test/left")
        self.images_right = load_images("./data/side2/test/right")

        self.keypoints_left = load_keypoints("./data/outputs/side2/2/train_labels.v001.003_test_left.analysis.h5")
        self.keypoints_right = load_keypoints("./data/outputs/side2/2/train_labels.v001.002_test_right.analysis.h5")
        
        # intrinsics = np.eye(3)
        # intrinsics[0, 0] = 500
        # intrinsics[1, 1] = 500
        # intrinsics[0, 2] = 640
        # intrinsics[1, 2] = 360
        extrinsics = np.eye(4)
        # extrinsics[2, 3] += 100

        # self.scene.setup_camera(self.intrinsics, extrinsics, 1280, 720, bbox)
        self.scene.setup_camera(self.intrinsics, extrinsics, 960, 540, bbox)

        self.window.add_child(self.scene)

        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            debug_menu = gui.Menu()
            debug_menu.add_item("Start", SpheresApp.MENU_RANDOM)
            debug_menu.add_separator()
            debug_menu.add_item("Quit", SpheresApp.MENU_QUIT)

            menu = gui.Menu()
            menu.add_menu("Debug", debug_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(SpheresApp.MENU_RANDOM,
                                               self._on_menu_random)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_QUIT,
                                               self._on_menu_quit)
        

        # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('graph', cv2.WINDOW_AUTOSIZE)

        self.graphs = get_graphs(self.vectors)
        

    def add_sphere(self):
        self._id += 1
        mat = rendering.MaterialRecord()
        mat.base_color = [
            random.random(),
            random.random(),
            random.random(), 1.0
        ]
        mat.shader = "defaultLit"
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.5)
        sphere.compute_vertex_normals()
        sphere.translate([
            10.0 * random.uniform(-1.0, 1.0), 10.0 * random.uniform(-1.0, 1.0),
            10.0 * random.uniform(-1.0, 1.0)
        ])
        self.scene.scene.add_geometry("sphere" + str(self._id), sphere, mat)

    def update_sphere(self):
        mat = rendering.MaterialRecord()
        mat.base_color = [1.0, 0, 0, 1.0]
        mat.shader = "defaultLit"
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(0.005, 0.275)
        cylinder.compute_vertex_normals()
        cylinder.transform(self.poses[self._id % len(self.poses)])
        if self._id > 0:
            self.scene.scene.remove_geometry("cylinder" + str(self._id - 1))
        self.scene.scene.add_geometry("cylinder" + str(self._id), cylinder, mat)
        self._id += 1

    def _on_menu_random(self):
        # This adds spheres asynchronously. This pattern is useful if you have
        # data coming in from another source than user interaction.
        def thread_main():
            for _ in range(0, len(self.poses) * 10):
                # We can only modify GUI objects on the main thread, so we
                # need to post the function to call to the main thread.
                gui.Application.instance.post_to_main_thread(
                    self.window, self.update_sphere)
                
                # self.scene.capture_screen_image()

                id = self._id % len(self.poses)

                output_string = f"X: {self.vectors[id][0].item(): .3f} - Y: {self.vectors[id][1].item(): .3f} - Z: {self.vectors[id][2].item(): .3f}"

                print(output_string)
                # sys.stdout.write(f"X: {self.vectors[id][0].item(): .3f} - Y: {self.vectors[id][1].item(): .3f} - Z: {self.vectors[id][2].item(): .3f}")
                # sys.stdout.flush()

                image = draw_image(self.images_left[id], self.images_right[id], self.keypoints_left[..., id], self.keypoints_right[..., id])
                
                image = cv2.resize(image, (960*2, 540))

                # fig, ax = plt.subplots()
                # ax.bar(np.arange(3), self.vectors[id], color="blue")

                # fig.canvas.draw()
                # X = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
                # plt.close()


                cv2.imshow('Stereo Pair', image)
                cv2.imshow('Directional Vector', self.graphs[id])
                cv2.waitKey(50)
                # time.sleep(0.05)

        # thread_main()

        threading.Thread(target=thread_main).start()

    def _on_menu_quit(self):
        gui.Application.instance.quit()


def main():
    gui.Application.instance.initialize()
    SpheresApp()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()