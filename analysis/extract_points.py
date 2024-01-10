import numpy as np
import math as m
import json

from tqdm import tqdm

from camera import Recording
from inference import Inference
from transforms import Project


if __name__ == "__main__":

    model_filepath = "./models/sleap_projects/models/7-points-3230831_174033.single_instance.n=300"

    # run_dir = "20231231_191639"
    run_dir = "20231231_191742"
    # run_dir = "20231231_191825"

    camera = Recording(f"./data/collection/{run_dir}")
    # camera = Recording("./data/collection/20231231_191742")
    # camera = Recording("./data/collection/20231231_191825")

    outputs = []

    inference = Inference(model_filepath)
    project = Project(
        intrinsics=camera.intrinsics,
        baseline=camera.baseline,
    )

    for i in tqdm(range(len(camera))):
        image_left, image_right = camera()
        points_left, points_right = inference.stereo_inference(image_left, image_right)

        output = {"left": [], "right": []}

        for i in range(len(points_left)):
            if m.isnan(points_left[i, 0].item()):
                point_left_output_x = -1
                point_left_output_y = -1
            else:
                point_left_output_x = points_left[i, 0].item()
                point_left_output_y = points_left[i, 1].item()

            output["left"].append([point_left_output_x, point_left_output_y])

            if m.isnan(points_right[i, 0].item()):
                point_right_output_x = -1
                point_right_output_y = -1
            else:
                point_right_output_x = points_right[i, 0].item()
                point_right_output_y = points_right[i, 1].item()

            output["right"].append([point_right_output_x, point_right_output_y])

        outputs.append(output)

    with open(f"./analysis/outputs/{run_dir}/points.json", 'w') as fp:
        json.dump(outputs, fp)
        
    camera.left_cap.release()
    camera.right_cap.release()
            
