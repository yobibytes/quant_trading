#!/bin/bash

export ZONE="europe-west1-d"
export INSTANCE_NAME="fastai-eval-instance"

# start a instance
gcloud compute instances start --zone=$ZONE $INSTANCE_NAME

# check: https://console.cloud.google.com/compute/
gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080

# check: http://localhost:8080/tree

# stop instance: https://console.cloud.google.com/compute/instances
