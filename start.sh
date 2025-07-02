#!/usr/bin/env bash

set -e
#启动jupyter
echo "[startup] Starting Jupyter Notebook in /jupyter ..."
jupyter lab --allow-root &
sleep 5
#启动训练服务
echo "[startup] Starting Train And Export Service ..."
python platform_service.py &
echo "[start.sh] Both Services are running, Enering wait ..."
wait
