#!/bin/sh

docker run --rm -it -v $(pwd):/work -w /work -u $(id -u):$(id -g) las-mesh:latest bash
