#!/bin/bash

# source: https://aws.amazon.com/getting-started/tutorials/get-started-dlami/

# EC2 - instances - IPV4 Public IP
ssh -i ~/.ssh/id_rsa -L localhost:8888:localhost:8888 ubuntu@3.121.202.157