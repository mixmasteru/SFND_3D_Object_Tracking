#!/bin/bash
FOLDER=${1}
VERSION=4.1.0

docker stop opencv
docker build --tag opencv:$VERSION .
docker run -d \
    --name opencv \
    --rm \
    --device=/dev/dri:/dev/dri \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$FOLDER:/root/project:rw" \
    -p 127.0.0.1:2223:22 \
    opencv:$VERSION

export containerId=$(docker ps -fname=opencv -q)
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId`
#docker start $containerId

