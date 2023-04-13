import os
import sys
import subprocess

from tqdm import tqdm


if __name__ == "__main__":
    sides = ("left", "right")
    dirpath_input = "/home/casey/Uni/Hip-Angle/data/uni_data/test"
    dirpath_output = "/home/casey/Uni/Hip-Angle/data/uni_data/test_reencode"

    for subdirname in tqdm(os.listdir(dirpath_input)):
        for side in sides:
            filepath_input = f"{dirpath_input}/{subdirname}/{side}.avi"
            filepath_output = f"{dirpath_output}/{subdirname}_{side}.mp4"

            bashCommand = f'ffmpeg -y -i {filepath_input} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 {filepath_output}'
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()