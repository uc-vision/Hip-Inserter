import numpy as np


class Project(object):
    def __init__(self, intrinsics, baseline):
        self.intrinsics = intrinsics
        self.baseline = baseline
        self.focal = intrinsics[0, 0]

    def disparity_to_depth(self, focal, baseline, disparity):
        depth = baseline * focal / disparity
        return depth
    
    def depth_to_disparity(self, focal, baseline, depth):
        disparity = baseline * focal / depth
        return disparity

    def project(self, points_left, points_right):
        depths = self.disparity_to_depth(self.focal, self.baseline, points_left[..., 0] - points_right[..., 0])
        p = np.concatenate([points_left, np.ones((points_left.shape[0], 1))], axis=1)  # [Points, 3]
        P = (np.linalg.inv(self.intrinsics) @ (p.T * depths[None, :])).T  # [Points, 3]
        return P
    
    def unproject(self, P):
        p = (self.intrinsics @ P.T).T
        points_left = p[:, :2] / p[:, 2, None]
        points_right = p[:, :2] / p[:, 2, None]
        points_right[:, 0] = points_right[:, 0] - self.depth_to_disparity(self.focal, self.baseline, p[:, 2])
        return points_left, points_right
    

def points_to_vector(points):
    """ Difference between edge points """
    return points[:, 0] - points[:, -1]


def roll_pitch_angles(vector):
    vector_norm = vector / np.linalg.norm(vector)
    X, Y, Z = vector_norm[0], -vector_norm[1], vector_norm[2]
    roll = np.arctan2(X, Y)
    pitch = np.arctan2(Z, Y)
    return np.rad2deg(roll), np.rad2deg(pitch)


def axis_angles(vector):
    vector_norm = vector / np.linalg.norm(vector)
    pass

def euler_angles(vector):
    vector_norm = vector / np.linalg.norm(vector)
    pass