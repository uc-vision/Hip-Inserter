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
                        img=image_left_,
                        center=(int(pl[j, 0]), int(pl[j, 1])),
                        radius=0,
                        color=color,
                        thickness=3)
            
        return image_left_
    
class DrawDepth(object):
    def __init__(self):
        # self.project = project
        pass

    def __call__(self, image_left, depth_pixel, current):
        image_left_ = np.copy(image_left)

        if np.isnan(depth_pixel).any():
            return image_left_
        
        if not current:
            return image_left_

        image_left_ = cv2.circle(
                        img=image_left_,
                        center=(int(depth_pixel[0]), int(depth_pixel[1])),
                        radius=0,
                        color=[255, 255, 0],
                        thickness=8)
        
        return image_left_
    

class DrawValues(object):
    def __init__(self):
        pass

    def __call__(self, roll, pitch, depth, vector_current, depth_current):
        image_roll, image_pitch = self.draw_angles(roll, pitch, vector_current)
        image_depth = self.draw_depth(depth, depth_current)

        image_values = np.concatenate([image_roll, image_pitch, image_depth], axis=0)

        return image_values

    def draw_angles(self, roll, pitch, current):
        image_roll = np.ones((240, 480, 3), dtype=np.uint8) * 255
        image_pitch = np.ones((240, 480, 3), dtype=np.uint8) * 255

        if np.isnan(roll).any() or np.isnan(pitch).any():
            return image_roll, image_pitch
        
        if current:
            color = (0, 196, 0)
        else:
            color = (196, 0, 0)
                
        cv2.putText(img=image_roll,
            text=f"X: {round(roll)}",
            org=(10,200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            color=color,
            thickness=8,
            bottomLeftOrigin=False)
        
        cv2.putText(img=image_pitch,
            text=f"Z: {round(pitch)}",
            org=(10,200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            color=color,
            thickness=8,
            bottomLeftOrigin=False)
        
        return image_roll, image_pitch
    
    def draw_depth(self, depth, current):
        image_depth = np.ones((240, 480, 3), dtype=np.uint8) * 255

        if np.isnan(depth).any():
            return image_depth

        if current:
            color = (0, 196, 0)
        else:
            color = (196, 0, 0)

        cv2.putText(img=image_depth,
            text=f"d: {round(depth, 3)}",
            org=(10,200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            color=color,
            thickness=8,
            bottomLeftOrigin=False)
        
        return image_depth