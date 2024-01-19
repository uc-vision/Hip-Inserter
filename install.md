

```
docker run --gpus all -it --name hip-inserter --privileged --ipc=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=/home/casey/Projects/docker_mount,target=/mnt stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04
```

`apt-get update`

`apt-get install python3-pyqt5`

`python3 -m pip install tensorflow[and-cuda]`

`python3 -m pip install sleap[pypi]==1.3.3`

`python3 -m pip install hydra-core tqdm`

`python3 /usr/local/zed/get_python_api.py`