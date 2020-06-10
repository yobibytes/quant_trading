#!/bin/bash

export ZONE="europe-west1-d"
export INSTANCE_NAME="fastai-eval-instance"

# start a instance
gcloud compute instances stop --zone=$ZONE $INSTANCE_NAME


# stop instance: https://console.cloud.google.com/compute/instances
