#!/bin/bash

# within fastai-eval-instance
# git clone https://github.com/fastai/course-v3
cd ~/tutorials/fastai/course-v3
git checkout .
git pull
sudo /opt/anaconda3/bin/conda update conda
sudo /opt/anaconda3/bin/conda install -c fastai fastai