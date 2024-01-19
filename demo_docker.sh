#!/bin/bash

xhost +local:root

docker start hip-inserter
docker exec -it hip-inserter python3 /mnt/Hip-Inserter/demo.py
docker stop hip-inserter