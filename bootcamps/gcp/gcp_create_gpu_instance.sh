#!/bin/bash

# or "pytorch-latest-cpu" for non-GPU instances
export IMAGE_FAMILY="pytorch-latest-gpu"
# check https://cloud.google.com/compute/docs/gpus/#gpus-list
export ZONE="europe-west1-d"
export INSTANCE_NAME="fastai-eval-instance"
# budget: "n1-highmem-4"
export INSTANCE_TYPE="n1-highmem-8"

# budget: 'type=nvidia-tesla-k80,count=1'
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-p100,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible

# check: https://console.cloud.google.com/compute/

# stop instance: https://console.cloud.google.com/compute/instances