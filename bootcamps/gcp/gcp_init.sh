#!/bin/bash

# source: https://course.fast.ai/start_gcp.html
export ZONE="europe-west1-d"

# Create environment variable for correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"

# Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

# once the installation is done try to log in and set an active project
gcloud init

# Run `gcloud help config` to learn how to change individual settings
# Run `gcloud --help` to see the Cloud Platform services you can interact with. And run `gcloud help COMMAND` to get help on any gcloud command
# Run `gcloud topic --help` to learn about advanced features of the SDK like arg files and output formatting

# set compute/zone to Frankfurt, Germany (see https://cloud.google.com/compute/docs/regions-zones )
gcloud config set compute/zone $ZONE

# stop an instance
# https://console.cloud.google.com/compute/instances