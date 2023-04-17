import numpy as np
import cv2
import math as m
import random

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from multiprocessing import Process, Pipe

from transforms import roll_pitch_angles


class DrawPoints(object):
    def __init__(self, scale=1, multi=False, num_points=100):
        """ Draw detected point on left and right images and draw connecting lines to visualise disparity"""
        self.scale = scale
        self.multi = multi

        self.colors = []
        for i in range(num_points):
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            self.colors.append(color)

        self.conn1, self.conn2 = Pipe()

        self.p = Process(target=self.process_loop, args=())
        self.p.start()

    def process_loop(self):
        while True:
            (image_left, image_right, points_left, points_right) = self.conn1.recv()
            self.display(image_left, image_right, points_left, points_right)

    def send(self, image_left, image_right, points_left, points_right):
        if self.multi:
            self.conn2.send((image_left, image_right, points_left, points_right))
        else:
            self.display(image_left, image_right, points_left, points_right)

    def draw_points(self, image, points):
        image_points = np.copy(image)
        for i in range(points.shape[0]):
            x, y = points[i, 0], points[i, 1]

            if i >= len(self.colors):
                raise IndexError("colors list index out of range")
            
            if (not m.isnan(x)) and (not m.isnan(y)):
                image_points = cv2.circle(
                    img=image_points,
                    center=(int(x), int(y)),
                    radius=0,
                    color=self.colors[i],
                    thickness=8)
        return image_points
    
    def draw_stereo(self, image_left, image_right, points_left, points_right):

        image_left_points = self.draw_points(image_left, points_left)
        image_right_points = self.draw_points(image_right, points_right)

        image_line = np.concatenate([image_left_points, image_right_points], axis=1)

        for i in range(points_left.shape[0]):
            x_l, y_l = points_left[i, 0], points_left[i, 1]
            x_r, y_r = points_right[i, 0], points_right[i, 1]

            if i >= len(self.colors):
                raise IndexError("colors list index out of range")
            
            if (not m.isnan(x_l)) and (not m.isnan(y_l)) and (not m.isnan(x_r)) and (not m.isnan(y_r)):
                x_r += image_left.shape[1]

                image_line = cv2.line(
                    image_line,
                    (int(x_l), int(y_l)),
                    (int(x_r), int(y_r)),
                    color=self.colors[i],
                    thickness=2)
        return image_line
    
    def display(self, image_left, image_right, points_left, points_right):
        image_stereo = self.draw_stereo(image_left, image_right, points_left, points_right)
        image_stereo = cv2.cvtColor(image_stereo, cv2.COLOR_BGR2RGB)

        if self.scale != 1:
            x_resize = int(image_stereo.shape[1] * self.scale)
            y_resize = int(image_stereo.shape[0] * self.scale)
            image_stereo = cv2.resize(image_stereo, (x_resize, y_resize))

        cv2.imshow("Stereo Pair", image_stereo)
        cv2.waitKey(1)


class DrawLine(object):
    def __init__(self, scale=1, multi=False, num_points=100):
        """ Draw detected points along inserter, and superimpose estimated line """
        self.scale = scale
        self.multi = multi

        self.colors = []
        for i in range(num_points):
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            self.colors.append(color)

        self.conn1, self.conn2 = Pipe()

        self.p = Process(target=self.process_loop, args=())
        self.p.start()

    def process_loop(self):
        while True:
            (image, points) = self.conn1.recv()
            self.display(image, points)

    def send(self, image, points):
        if self.multi:
            self.conn2.send((image, points))
        else:
            self.display(image, points)

    def draw_points(self, image, points):
        image_points = np.copy(image)

        x1, y1 = points[0, 0], points[0, 1]
        x2, y2 = points[-1, 0], points[-1, 1]
        
        if (not m.isnan(x1)) and (not m.isnan(y1)):
            cv2.circle(
                img=image_points,
                center=(int(x1), int(y1)),
                radius=0,
                color=[255, 0, 0],
                thickness=10)
        else:
            return
        
        if (not m.isnan(x2)) and (not m.isnan(y2)):
            cv2.circle(
                img=image_points,
                center=(int(x2), int(y2)),
                radius=0,
                color=[0, 255, 0],
                thickness=10)
        else:
            return
                
        cv2.line(
            image_points,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=[255, 255, 0],
            thickness=4)


        return image_points
    
    def display(self, image, points):
        image_line = self.draw_points(image, points)
        image_line = cv2.cvtColor(image_line, cv2.COLOR_BGR2RGB)

        if self.scale != 1:
            x_resize = int(image_line.shape[1] * self.scale)
            y_resize = int(image_line.shape[0] * self.scale)
            image_line = cv2.resize(image_line, (x_resize, y_resize))

        cv2.imshow("Camera", image_line)
        cv2.waitKey(1)


class DrawAngles(object):
    def __init__(self, multi=False):
        """ Calculate the roll and pitch of inserter, and display angle in degrees """
        self.multi = multi

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.conn1, self.conn2 = Pipe()

        self.p = Process(target=self.process_loop, args=())
        self.p.start()

    def process_loop(self):
        while True:
            vector = self.conn1.recv()
            self.display(vector)

    def send(self, vector):
        if self.multi:
            self.conn2.send(vector)
        else:
            self.display(vector)

    def display(self, vector):
        if np.isnan(vector).any():
            return
        
        roll, pitch = roll_pitch_angles(vector)

        image_roll = np.ones((480, 640, 3), dtype=np.uint8) * 255
        image_pitch = np.ones((480, 640, 3), dtype=np.uint8) * 255

        cv2.putText(img=image_roll,
                    # text=f"Roll (θ): {roll:.3f}",
                    text=f"X: {abs(round(roll))}",
                    org=(10,400),
                    fontFace=self.font,
                    fontScale=5,
                    color=(0, 0, 128),
                    thickness=8,
                    bottomLeftOrigin=False)
        
        cv2.putText(img=image_pitch,
                    # text=f"Pitch (φ): {pitch:.3f}",
                    text=f"Z: {abs(round(pitch))}",
                    org=(10,400),
                    fontFace=self.font,
                    fontScale=5,
                    color=(128, 0, 0),
                    thickness=8,
                    bottomLeftOrigin=False)

        image_angles = np.concatenate([image_roll, image_pitch], axis=1)

        cv2.putText(img=image_angles,
                    # text=f"Roll (θ): {roll:.3f}",
                    text=f"Inserter Angles:",
                    org=(300,100),
                    fontFace=self.font,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=8,
                    bottomLeftOrigin=False)

        cv2.imshow("Inserter Angles", image_angles)
        cv2.waitKey(1)


class Visualiser3D:
    def __init__(self, intrinsics):
        """ Note: Non-Blocking 3D Visualiser currently unsolved (Do not use)"""

        gui.Application.instance.initialize()

        self.intrinsics = intrinsics

        self._id = 0

        self.window = gui.Application.instance.create_window("Spatial Rendering", 1280, 720)
        
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.scene.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self.scene.scene.show_axes(True)

        self.scene.setup_camera(self.intrinsics, np.eye(4), 1280, 720, bbox)

        gui.Application.instance.run()

    def cylinder_pose(self, P1, P2):
        midpoint = (P1 + P2) / 2
        translation = np.eye(4)
        translation[:3, 3] = midpoint

        distance = np.linalg.norm(P1 - P2)
        direction_vector = (P1 - P2) / distance
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
        pose = translation @ rotation_matrix
        return pose

    def update_cylinder(self, P1, P2):
        if np.isnan(P1).any() or np.isnan(P2).any():
            return

        mat = rendering.MaterialRecord()
        mat.base_color = [1.0, 0, 0, 1.0]
        mat.shader = "defaultLit"
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(0.005, 0.275)
        cylinder.transform(self.cylinder_pose(P1, P2))
        cylinder.compute_vertex_normals()

        if self._id > 0:
            self.scene.scene.remove_geometry("cylinder" + str(self._id - 1))
        self.scene.scene.add_geometry("cylinder" + str(self._id), cylinder, mat)
        self._id += 1