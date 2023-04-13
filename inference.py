import numpy as np

from sleap.nn.inference import load_model


class Inference(object):
    def __init__(self, model_filepath):
        # self.model = load_model(model_filepath)
        self.model = load_model(model_filepath, resize_input_layer=False)

    def stereo_inference(self, image_left, image_right):
        image_batch = np.stack([image_left, image_right], axis=0)
        out = self.model.inference_model(image_batch)
        points = out["instance_peaks"].numpy()[:, 0, :, :]  # [Batch, Points, 2]
        points_left, points_right = points[0], points[1]  # [Points, 2]
        return points_left, points_right
