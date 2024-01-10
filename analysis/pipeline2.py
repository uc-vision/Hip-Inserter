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

from visualizers import DrawPoints, DrawLine, DrawAngles
from camera import Recording
from inference import Inference
from transforms import Project


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
    points_left_new = deepcopy(points_left)
    points_right_new = deepcopy(points_right)

    for j in range(len(points_left)):
        if m.isnan(points_left[j, 0]) or m.isnan(points_right[j, 0]):
            continue
        vertical_delta = abs(points_left[j, 1] - points_right[j, 1])
        if vertical_delta > pixel_delta:
            points_left_new[j, 0] = m.nan
            points_left_new[j, 1] = m.nan
            points_right_new [j, 0] = m.nan
            points_right_new [j, 1] = m.nan

    return points_left_new, points_right_new





if __name__ == "__main__":

    model_filepath = "./models/sleap_projects/models/7-points-3230831_174033.single_instance.n=300"

    # camera = Camera()
    camera = Recording("./data/collection/20231231_191639")
    # camera = Recording("./data/collection/20231231_191742")
    # camera = Recording("./data/collection/20231231_191825")

    inference = Inference(model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    # left_buffer = Buffer2D(0.2, 1)
    left_buffers = Buffers2D(0.1, 2, 5)
    right_buffers = Buffers2D(0.1, 2, 5)
    
    t = 100

    initial = np.array([
        [0, 0.03, 0.07, 0.16, 0.28],
        [0.03, 0, 0.04, 0.13, 0.25],
        [0.07, 0.04, 0, 0.09, 0.21],
        [0.16, 0.13, 0.09, 0, 0.12],
        [0.28, 0.25, 0.21, 0.12, 0],
        ])
    initial = initial / 0.28

    for i in tqdm(range(len(camera))):

        t += 1 / 30
        image_left, image_right = camera()

        points_left, points_right = inference.stereo_inference(image_left, image_right)
        points_left = points_left[:5, :]
        points_right = points_right[:5, :]

        points_left, points_right = horozontal_check(points_left, points_right, 2)