#!/usr/bin/env bash
set -euo pipefail
DEPLOY_HOST=$1
rsync -av --delete requirements.txt train.py ccc.py ccc.sh ${DEPLOY_HOST}:/home/rob/ccc/
rsync -av --delete images ${DEPLOY_HOST}:/home/rob/ccc/
