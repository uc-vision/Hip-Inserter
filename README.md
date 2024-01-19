# Hip-Inserter
The goal of this project is to estimate the angles/orientation of a orthopaedic hip inserter using a pipeline of computer vision methods.

![Angle Estimation](./images/intro.gif)

## Install
To install this project we recommend using using the following docker container and flags. 

`docker run --gpus all -it --name hip-inserter --privileged --ipc=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=/home/user/Documents/docker_mount,target=/mnt stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04`

### Docker Container
Run the following install commands inside the docker container.

`apt-get update`

`apt-get install python3-pyqt5`

`python3 -m pip install tensorflow[and-cuda]==2.8.4`

`python3 -m pip install sleap[pypi]==1.3.3`

`python3 -m pip install hydra-core tqdm`

`python3 /usr/local/zed/get_python_api.py`


### Run Docker

`chmod +x ./demo_docker.sh`

`./demo_docker.sh`


### Setup Desktop

`cp ./demo.desktop ~/Desktop/demo.desktop`

`chmod +x ~/Desktop/demo.desktop`

`gio set ~/Desktop/demo.desktop metadata::trusted true`
