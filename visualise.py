import numpy as np
import cv2

class DrawVector(object):
    def __init__(self, project, n_points):
        self.project = project
        self.n_points = n_points

    def __call__(self, image_left, mean, vector, current):
        image_left_ = np.copy(image_left)

        if np.isnan(mean).any() or np.isnan(vector).any():
            return image_left_

        p3d = mean[None, :] + vector[None, :] * np.linspace(-2, 2, self.n_points)[:, None]
        pl, pr = self.project.unproject(p3d)

        if current:
            color = [0, 255, 0]
        else:
            color = [255, 0, 0]

        for j in range(self.n_points):
            image_left_ = cv2.circle(
                        img=image_left,
                        center=(int(pl[j, 0]), int(pl[j, 1])),
                        radius=0,
                        color=color,
                        thickness=3)
            
        return image_left_