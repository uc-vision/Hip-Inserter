import numpy as np
import math as m
import json
import cv2
import matplotlib.pyplot as plt
import random

from copy import deepcopy
from tqdm import tqdm
from sklearn import datasets, decomposition
from time import time

from camera import Recording
from inference import Inference
from transforms import Project, roll_pitch_angles


class Buffer2D(object):
    def __init__(self, time_delta, pixel_delta):
        self.time_delta = time_delta
        self.pixel_delta = pixel_delta
        self.buffer = {}

    def __call__(self, array, t):
        self.add(array, t)
        self.update(t)
        return self.output()

    def add(self, array, t):
        if np.isnan(array).any():
            return None
        self.buffer[t] = array

    def update(self, t):
        old_keys = []
        for key, val in self.buffer.items():
            if t - key > self.time_delta:
                old_keys.append(key)
        for key in old_keys:
            self.buffer.pop(key)

    def output(self):
        if len(self.buffer) == 0:
            return None

        max_key = max(self.buffer.keys())

        subset = {}
        scores = {}

        for key1, val1 in self.buffer.items():
            scores[key1] = []
            subset[key1] = []
            for key2, val2 in self.buffer.items():
                dist = np.linalg.norm(val2 - val1)
                if dist > self.pixel_delta:
                    continue
                scores[key1].append((self.time_delta - (max_key - key2)) / self.time_delta)
                subset[key1].append(val2)

        best_key = None
        best_score = 0
        for key, val in self.buffer.items():
            if sum(scores[key]) > best_score:
                best_score = sum(scores[key])
                best_key = key

        answer = np.zeros([2, ])

        for i in range(len(scores[best_key])):
            answer += subset[best_key][i] * scores[best_key][i] / best_score

        return answer
    
class Buffers2D(object):
    def __init__(self, time_delta, pixel_delta, num_buffers):
        self.time_delta = time_delta
        self.pixel_delta = pixel_delta
        self.num_buffers = num_buffers

        self.buffers = []
        for i in range(self.num_buffers):
            self.buffers.append(Buffer2D(self.time_delta, self.pixel_delta))

    def __call__(self, array, t):
        self.add(array, t)
        self.update(t)
        return self.output()

    def add(self, array, t):
        for i in range(self.num_buffers):
            self.buffers[i].add(array[i], t)

    def update(self, t):
        for i in range(self.num_buffers):
            self.buffers[i].update(t)
            
    def output(self):
        answers = np.zeros([self.num_buffers, 2])
        for i in range(self.num_buffers):
            answer = self.buffers[i].output()
            if answer is not None:
                answers[i] = self.buffers[i].output()
            else:
                answers[i] = np.array([np.nan, np.nan])
        return answers


def horozontal_check(points_left, points_right, pixel_delta):
    for j in range(len(points_left)):
        if m.isnan(points_left[j, 0]) or m.isnan(points_right[j, 0]):
            continue
        vertical_delta = abs(points_left[j, 1] - points_right[j, 1])
        if vertical_delta > pixel_delta:
            points_left[j, 0] = m.nan
            points_left[j, 1] = m.nan
            points_right[j, 0] = m.nan
            points_right[j, 1] = m.nan


def geometry_check(points_3d, dist_delta):
    dists_true = np.array([
        [0, 0.03, 0.07, 0.16, 0.28],
        [0.03, 0, 0.04, 0.13, 0.25],
        [0.07, 0.04, 0, 0.09, 0.21],
        [0.16, 0.13, 0.09, 0, 0.12],
        [0.28, 0.25, 0.21, 0.12, 0],
        ])
    
    dists_pred = np.linalg.norm(points_3d[None, :, :] - points_3d[:, None, :], axis=-1)
    
    mask = np.abs(dists_pred - dists_true) < dist_delta

    idx = np.argmax(np.sum(mask, axis=0))

    for j in range(points_3d.shape[0]):
        if mask[j, idx] == False:
            points_3d[j, :] = np.nan

# def geometry_check_2d(points_left, points_right, dist_de)


def principle_axis(points_3d):
    pca = decomposition.PCA(3)
    pca.fit(points_3d_sub)

    line_set = pca.mean_[None, :] + pca.components_[None, 0] * np.linspace(-0.2, 0.2, 500)[:, None]

    p = project.unproject(line_set)


def get_zero_position(points_3d, norm_vector):
    # dists_true = np.array([
    #     [0, 0.03, 0.07, 0.16, 0.28],
    #     [0.03, 0, 0.04, 0.13, 0.25],
    #     [0.07, 0.04, 0, 0.09, 0.21],
    #     [0.16, 0.13, 0.09, 0, 0.12],
    #     [0.28, 0.25, 0.21, 0.12, 0],
    #     ])
    
    pos_true = [0, 0.03, 0.07, 0.16, 0.28]
    # pos_true = [0.28, 0.16, 0.07, 0.03, 0]

    pos_zero = np.zeros((0, 3))

    for j in range(points_3d.shape[0]):
        if not (np.isnan(points_3d[j, :]).any()):
            pos_zero = np.concatenate([pos_zero, points_3d[j] - pos_true[j] * norm_vector])

    pos_zero_avg = np.mean(pos_zero, axis=0)

    return pos_zero_avg

    
    # dists_pred = np.linalg.norm(points_3d[None, :, :] - points_3d[:, None, :], axis=-1)


def closest_point_vector(q, v1, v2):
    ''' calculate interpolated 't' value along vector between points v1 and v2 '''
    n = (v1[0] - q[:, 0])*(v2[0] - v1[0]) + (v1[1] - q[:, 1])*(v2[1] - v1[1]) + (v1[2] - q[:, 2])*(v2[2] - v1[2])
    d = (v2[0] - v1[0])**2 + (v2[1] - v1[1])**2 + (v2[2] - v1[2])**2
    t = -n / d
    return t

def pick_best(points_3d, max_residual, min_length):
    
    pca = decomposition.PCA(3)
    pca.fit(points_3d)

    for i in range(len(points_3d)):
        pass


def line_2d_check(points_left, points_right, max_var):

    points_left_sub = np.zeros((0, 2))
    points_right_sub = np.zeros((0, 2))

    for j in range(points_left.shape[0]):
        if np.isnan(points_left[j]).any() or np.isnan(points_right[j]).any():
            continue
        points_left_sub = np.concatenate([points_left_sub, points_left[None, j]], axis=0)
        points_right_sub = np.concatenate([points_right_sub, points_right[None, j]], axis=0)

    if points_left_sub.shape[0] < 3:
        output = np.zeros(points_left.shape)
        output[...] = np.nan
        return output, output
    
    pca_left = decomposition.PCA(2)
    pca_left.fit(points_left_sub)
    pca_left.explained_variance_ratio_

    pca_right = decomposition.PCA(2)
    pca_right.fit(points_right_sub)
    pca_right.explained_variance_ratio_

    if pca_left.explained_variance_ratio_[-1] > max_var or pca_right.explained_variance_ratio_[-1] > max_var:
        output = np.zeros(points_left.shape)
        output[...] = np.nan
        return output, output
        
    return points_left, points_right
    


class MovingAvg(object):
    def __init__(self, max_time_delta):
        self.max_time_delta = max_time_delta

        self.values = np.zeros([0])
        self.times = np.zeros([0])

    def __call__(self, value, time_):
        self.update(value, time_)
        self.discard(time_)
        return self.output()

    def update(self, value, time_):
        self.values = np.concatenate([self.values, np.array([value])])
        self.times = np.concatenate([self.times, np.array([time_])])

    def discard(self, time_):
        self.values = self.values[(time_ - self.times) <= self.max_time_delta]
        self.times = self.times[(time_ - self.times) <= self.max_time_delta]

    def output(self):
        return np.mean(self.values)



if __name__ == "__main__":

    model_filepath = "./models/sleap_projects/models/7-points-3230831_174033.single_instance.n=300"

    # camera = Camera()
    # camera = Recording("./data/collection/20231231_191639")
    # camera = Recording("./data/collection/20231231_191742")
    # camera = Recording("./data/collection/20231231_191825")
    # camera = Recording("./data/collection/20240107_183340")
    # camera = Recording("./data/collection/20240107_183449")
    camera = Recording("./data/collection/20240107_183743")
    # camera = Recording("./data/collection/20240107_183830")
    # camera = Recording("./data/collection/20240107_183916")
    # camera = Recording("./data/collection/20240107_183955")

    inference = Inference(model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    fourcc = cv2.VideoWriter_fourcc(*'h264')
    # cap = cv2.VideoWriter("./analysis/outputs/20231231_191639/processed_output_keep.mp4", fourcc, 20.0, (1280+360, 720))
    # cap = cv2.VideoWriter("./analysis/outputs/20231231_191742/processed_output.mp4", fourcc, 20.0, (1920, 720))
    # cap = cv2.VideoWriter("./analysis/outputs/20231231_191825/processed_output.mp4", fourcc, 20.0, (1920, 720))

    # cap = cv2.VideoWriter("./analysis/outputs/20240107_183340/20240107_183340_proc.mp4", fourcc, 20.0, (1280+360, 720))
    cap = cv2.VideoWriter("./analysis/outputs/20240107_183743/20240107_183743_proc_4.mp4", fourcc, 20.0, (1280+360, 720))
    # cap = cv2.VideoWriter("./analysis/outputs/20240107_183955/20240107_183955_proc.mp4", fourcc, 10.0, (1280+360, 720))

    # left_buffer = Buffer2D(0.2, 1)
    left_buffers = Buffers2D(0.1, 2, 5)
    right_buffers = Buffers2D(0.1, 2, 5)
    
    t = 100

    roll_avg = MovingAvg(0.1)
    pitch_avg = MovingAvg(0.5)

    depth_avg = MovingAvg(0.1)

    current_once = False

    for i in tqdm(range(len(camera))):

        t += 1 / 30
        image_left, image_right = camera()

        points_left, points_right = inference.stereo_inference(image_left, image_right)
        points_left = points_left[:5, :]
        points_right = points_right[:5, :]

        horozontal_check(points_left, points_right, 2)

        points_left = left_buffers(points_left, t)
        points_right = right_buffers(points_right, t)

        points_left, points_right = line_2d_check(points_left, points_right, 0.001)

        # left_buffers.add(points_left, t)
        # left_buffers.update(t)
        # points_left = left_buffers.output()

        # right_buffers.add(points_right, t)
        # right_buffers.update(t)
        # points_right = right_buffers.output()

        points_3d = project.project(points_left, points_right)
        print(points_3d[:, 2])

        geometry_check(points_3d, 0.02)

        points_left_sub = np.zeros((0, 2))
        points_right_sub = np.zeros((0, 2))
        points_3d_sub = np.zeros((0, 3))

        for j in range(points_3d.shape[0]):
            if np.isnan(points_3d[j]).any():
                continue
            points_left_sub = np.concatenate([points_left_sub, points_left[None, j]], axis=0)
            points_right_sub = np.concatenate([points_right_sub, points_right[None, j]], axis=0)
            points_3d_sub = np.concatenate([points_3d_sub, points_3d[None, j]], axis=0)

        # print(points_3d_sub.shape)
            
        image_points = image_left

        image_angles = np.ones((720, 360, 3), dtype=np.uint8) * 255

        current = False

        if points_3d_sub.shape[0] >= 3:
            points_dist_sub = np.linalg.norm(points_3d_sub[None, :, :] - points_3d_sub[:, None, :], axis=-1)
            if np.amax(points_dist_sub) > 0.10:

                pca = decomposition.PCA(3)
                pca.fit(points_3d_sub)
                # print(pca.explained_variance_ratio_)

                # pos_zero = get_zero_position(points_3d, pca.components_[None, 0])
                axis_vector = np.copy(pca.components_[0])
                axis_vector = -axis_vector
                roll, pitch = roll_pitch_angles(axis_vector)

                roll = roll_avg(roll, t)
                pitch = pitch_avg(pitch, t)
                if not np.isnan(pca.mean_[2]):
                    depth = depth_avg(pca.mean_[2], t)

                image_roll = np.ones((240, 360, 3), dtype=np.uint8) * 255
                image_pitch = np.ones((240, 360, 3), dtype=np.uint8) * 255
                image_depth = np.ones((240, 360, 3), dtype=np.uint8) * 255

                cv2.putText(img=image_roll,
                    # text=f"Roll (θ): {roll:.3f}",
                    text=f"X: {round(roll)}",
                    org=(10,200),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 196, 0),
                    thickness=8,
                    bottomLeftOrigin=False)
        
                cv2.putText(img=image_pitch,
                    # text=f"Pitch (φ): {pitch:.3f}",
                    text=f"Z: {round(pitch)}",
                    org=(10,200),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 196, 0),
                    thickness=8,
                    bottomLeftOrigin=False)
                
                cv2.putText(img=image_depth,
                    # text=f"Pitch (φ): {pitch:.3f}",
                    text=f"d: {round(depth, 1)}",
                    org=(10,200),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 196, 0),
                    thickness=8,
                    bottomLeftOrigin=False)
                
                image_angles = np.concatenate([image_roll, image_pitch, image_depth], axis=0)

                line_set = pca.mean_[None, :] + pca.components_[None, 0] * np.linspace(-2, 2, 500)[:, None]
                # line_set = pos_zero + pca.components_[None, 0] * np.linspace(0-0.135, 0.28+0.035, 500)[:, None]
                # line_set = pos_zero + pca.components_[None, 0] * np.linspace(0, 0.28, 500)[:, None]

                pl, pr = project.unproject(line_set)

                for j in range(500):
                    image_left = cv2.circle(
                                img=image_left,
                                center=(int(pl[j, 0]), int(pl[j, 1])),
                                radius=0,
                                color=[0, 255, 0],
                                thickness=3)
                    
                    # image_right = cv2.circle(
                    #             img=image_right,
                    #             center=(int(pr[j, 0]), int(pr[j, 1])),
                    #             radius=0,
                    #             color=[255, 0, 255],
                    #             thickness=3)

                current_once = True
                current = True

        if not current and current_once:
            for j in range(500):
                image_left = cv2.circle(
                                    img=image_left,
                                    center=(int(pl[j, 0]), int(pl[j, 1])),
                                    radius=0,
                                    color=[255, 0, 0],
                                    thickness=3)
                
                cv2.putText(img=image_roll,
                    # text=f"Roll (θ): {roll:.3f}",
                    text=f"X: {round(roll)}",
                    org=(10,200),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(196, 0, 0),
                    thickness=8,
                    bottomLeftOrigin=False)
        
                cv2.putText(img=image_pitch,
                    # text=f"Pitch (φ): {pitch:.3f}",
                    text=f"Z: {round(pitch)}",
                    org=(10,200),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(196, 0, 0),
                    thickness=8,
                    bottomLeftOrigin=False)
                
                cv2.putText(img=image_depth,
                    # text=f"Pitch (φ): {pitch:.3f}",
                    text=f"d: {round(depth, 1)}",
                    org=(10,200),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(196, 0, 0),
                    thickness=8,
                    bottomLeftOrigin=False)
                
                image_angles = np.concatenate([image_roll, image_pitch, image_depth], axis=0)
                    
            

        # for point_left_sub in points_left_sub:
        #     if point_left_sub[0] != 0 and point_left_sub[1] != 0:
        #         image_points = cv2.circle(
        #                     img=image_points,
        #                     center=(int(point_left_sub[0]), int(point_left_sub[1])),
        #                     radius=0,
        #                     color=[0, 255, 255],
        #                     thickness=8)

        # if answer is not None:
        # # if not(np.isnan(points_left[1, :]).any()):
        #     image_points = cv2.circle(
        #                 img=image_points,
        #                 center=(int(answer[0]), int(answer[1])),
        #                 # center=(int(points_left[1, 0]), int(points_left[1, 1])),
        #                 radius=0,
        #                 color=[255, 255, 0],
        #                 thickness=8)
                    
        # image_points = np.concatenate([image_left, image_right], axis=1)
        # image_points = np.concatenate([image_left], axis=1)

        # print(image_points.shape, image_angles.shape)

        image = np.concatenate([image_left, image_angles], axis=1)

        # x_resize = int(image.shape[1] * 0.75)
        # y_resize = int(image.shape[0] * 0.75)
        # image = cv2.resize(image, (x_resize, y_resize))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cap.write(np.ascontiguousarray(image))

        cv2.imshow("image", image)
        cv2.waitKey(1)

    cap.release()


    cv2.destroyAllWindows()