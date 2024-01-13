from typing import Any
import numpy as np

from sklearn import  decomposition


def reduce_to_nonan_array(array):
    array_ = np.copy(array)
    mask = ~np.isnan(np.mean(array, axis=1))[:, None]
    mask = np.broadcast_to(mask, array_.shape)
    array_sub = np.reshape(array_[mask], (np.sum(mask[:, 0]), array_.shape[1]))
    return array_sub


def combination_masks(num_bools, min_true):
    masks = []
    for i in range(num_bools, min_true-1, -1):
        for j in range(2**num_bools):
            b = format(j, f'0{num_bools}b')
            mask = np.zeros(num_bools, bool)
            for k in range(num_bools):
                mask[k] = int(b[k])
            if np.sum(mask) == i:
                masks.append(mask)
    return masks

def closest_point_vector(q, v1, v2):
    ''' calculate interpolated 't' value along vector between points v1 and v2 '''
    n = (v1[0] - q[:, 0])*(v2[0] - v1[0]) + (v1[1] - q[:, 1])*(v2[1] - v1[1]) + (v1[2] - q[:, 2])*(v2[2] - v1[2])
    d = (v2[0] - v1[0])**2 + (v2[1] - v1[1])**2 + (v2[2] - v1[2])**2
    t = -n / d
    return t


class HorozontalCheck(object):

    def __init__(self, pixel_delta_max):
        self.pixel_delta_max = pixel_delta_max

    def __call__(self, points_left, points_right):

        points_left_ = np.copy(points_left)
        points_right_ = np.copy(points_right)

        vertical_delta = abs(points_left_[:, 1] - points_right_[:, 1])

        mask = vertical_delta < self.pixel_delta_max
        mask_broadcast = np.broadcast_to(mask[:, None], points_left_.shape)
        
        points_left_[~mask_broadcast] = np.nan
        points_right_[~mask_broadcast] = np.nan

        return points_left_, points_right_


class PrincipleAxisVarCheck(object):

    def __init__(self, var_max, points_min):
        self.var_max = var_max
        self.points_min = points_min

    def single(self, points, mask):
        points_ = np.copy(points)
        for i in range(mask.shape[0]):
            if not mask[i]:
                points_[i, :] = np.nan

        if self.count_notnans(points_) < self.points_min:
            points_[...] = np.nan
            return points_

        points_sub = reduce_to_nonan_array(points_)

        pca = decomposition.PCA(2)
        pca.fit(points_sub)

        if pca.explained_variance_ratio_[1] > self.var_max:
            points_[...] = np.nan
            return points_
        
        return points_
    
    def __call__(self, points):
        masks = combination_masks(5, self.points_min)
        for mask in masks:
            points_ = self.single(points, mask)
            if self.count_notnans(points_) >= self.points_min:
                return points_
        points_[...] = np.nan
        return points_

    def both(self, points_left, points_right):
        points_left_ = self(points_left)
        points_right_ = self(points_right)

        if np.isnan(points_left_).all() or np.isnan(points_right_).all():
            points_left_[...] = np.nan
            points_right_[...] = np.nan

        return points_left_, points_right_

    def count_notnans(self, points):
        return np.sum(~np.isnan(np.mean(points, axis=1)))
    

class MovingAvg(object):
    def __init__(self, time_delta_max):
        self.time_delta_max = time_delta_max

        self.values = np.zeros([0])
        self.times = np.zeros([0])

    def __call__(self, value, t):
        if not np.isnan(value):
            self.update(value, t)
            self.discard(t)
        return self.output()

    def update(self, value, t):
        self.values = np.concatenate([np.array([value]), self.values])
        self.times = np.concatenate([np.array([t]), self.times])

    def discard(self, t):
        time_mask = (t - self.times) <= self.time_delta_max
        self.values = self.values[time_mask]
        self.times = self.times[time_mask]

    def output(self):
        if len(self.values) == 0:
            return np.nan
        return np.mean(self.values)
    

class Buffer(object):
    def __init__(self, time_delta_max, pixel_delta_max):
        self.time_delta_max = time_delta_max
        self.pixel_delta_max = pixel_delta_max

        self.values = np.zeros([0, 2])
        self.times = np.zeros([0])

    def __call__(self, value, t):
        self.update(value, t)
        self.discard(t)
        return self.output()

    def update(self, value, t):
        if np.isnan(value).any():
            return
        self.values = np.concatenate([np.array([value]), self.values], axis=0)
        self.times = np.concatenate([np.array([t]), self.times])

    def discard(self, t):
        if self.values.shape[0] == 0:
            return
        time_mask = (t - self.times) <= self.time_delta_max
        time_mask_broadcast = np.broadcast_to(time_mask[:, None], self.values.shape)
        self.values = np.reshape(self.values[time_mask_broadcast], [np.sum(time_mask), 2])
        self.times = self.times[time_mask]
        
        if self.values.shape[0] == 0:
            return
        pixel_mask = np.linalg.norm(self.values[0, None, :] - self.values, axis=1) <= self.pixel_delta_max
        pixel_mask_broadcast = np.broadcast_to(pixel_mask[:, None], self.values.shape)
        self.values = np.reshape(self.values[pixel_mask_broadcast], [np.sum(pixel_mask), 2])
        self.times = self.times[pixel_mask]

    def output(self):
        if self.values.shape[0] == 0:
            output = np.zeros([1, 2])
            output[...] = np.nan
            return output
        return np.nanmean(self.values, axis=0)
    

class Buffers(object):
    def __init__(self, time_delta_max, pixel_delta_max, num_buffers):
        self.num_buffers = num_buffers
        self.buffers = []
        for i in range(num_buffers):
            self.buffers.append(Buffer(time_delta_max, pixel_delta_max))

    def __call__(self, values, t):
        outputs = np.zeros([self.num_buffers, 2])
        for i in range(self.num_buffers):
            outputs[i, :] = self.buffers[i](values[i], t)
        return outputs


class GeometryCheck(object):
    def __init__(self, distance_delta_max, points_min):
        self.distance_delta_max = distance_delta_max
        self.points_min = points_min
        self.distance_true = np.array([
            [0, 0.03, 0.07, 0.16, 0.28],
            [0.03, 0, 0.04, 0.13, 0.25],
            [0.07, 0.04, 0, 0.09, 0.21],
            [0.16, 0.13, 0.09, 0, 0.12],
            [0.28, 0.25, 0.21, 0.12, 0],
            ])
        

    def __call__(self, points_3d):
        masks = combination_masks(5, self.points_min)
        for mask in masks:
            points_3d_ = self.single(points_3d, mask)
            if self.count_notnans(points_3d_) >= self.points_min:
                return points_3d_
        points_3d_[...] = np.nan
        return points_3d_
        

    def single(self, points_3d, mask):
        points_3d_ = np.copy(points_3d)
        for i in range(mask.shape[0]):
            if not mask[i]:
                points_3d_[i, :] = np.nan

        if self.count_notnans(points_3d_) < self.points_min:
            points_3d_[...] = np.nan

        distance_pred = np.linalg.norm(points_3d_[None, :, :] - points_3d_[:, None, :], axis=-1)
        distance_delta = np.abs(distance_pred - self.distance_true)

        if np.nanmax(distance_delta) > self.distance_delta_max:
            points_3d_[...] = np.nan

        return points_3d_
    
    def count_notnans(self, points):
        return np.sum(~np.isnan(np.mean(points, axis=1)))


class DistanceCheck(object):
    def __init__(self, distance_min):
        self.distance_min = distance_min

    def __call__(self, points_3d):
        points_3d_ = np.copy(points_3d)
        points_dist = np.nanmax(np.linalg.norm(points_3d_[None, :, :] - points_3d_[:, None, :], axis=-1))
        if points_dist < self.distance_min:
            points_3d_[...] = np.nan
        return points_3d_


class Points2Vector(object):
    def __init__(self):
        self.mean = np.zeros([3])
        self.mean[...] = np.nan
        self.vector = np.zeros([3])
        self.vector[...] = np.nan
        self.current = False

    def __call__(self, points_3d):
        if np.isnan(points_3d).all():
            self.current = False
        else:
            self.current = True
            self.update(points_3d)
        return self.get_vector()

    def update(self, points_3d):
        points_3d_sub = reduce_to_nonan_array(points_3d)
        pca = decomposition.PCA(3)
        pca.fit(points_3d_sub)

        mean = np.copy(pca.mean_)
        vector = -np.copy(pca.components_[0])

        ord = closest_point_vector(points_3d_sub, mean, mean + vector)
        if ord[0] < ord[-1]:
            vector = -vector

        self.vector = vector
        self.mean = mean

    def get_vector(self):
        return self.mean, self.vector
    
    def get_current(self):
        return self.current
    

class DepthEstimate(object):
    def __init__(self, point_idx):
        self.point_idx = point_idx
        self.current = False
        self.depth = np.nan
        self.depth_pixel = np.zeros(2)
        self.depth_pixel[...] = np.nan

    def __call__(self, points_3d, points_left):
        if np.isnan(points_3d[self.point_idx, :]).any():
            self.current = False
        else:
            self.current = True
            self.update(points_3d, points_left)
        return self.get_depth()

        
    def update(self, points_3d, points_left):
        self.depth = points_3d[self.point_idx, 2]
        self.depth_pixel = points_left[self.point_idx, :]

    def get_depth(self):
        return self.depth, self.depth_pixel

    def get_current(self):
        return self.current
        
