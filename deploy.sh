#!/usr/bin/env bash
set -euo pipefail
DEPLOY_HOST=$1
rsync -av setup.sh lab_classifications.csv requirements.txt train.py ccc.py ${DEPLOY_HOST}:/home/rob/ccc/
rsync -av images ${DEPLOY_HOST}:/home/rob/ccc/
