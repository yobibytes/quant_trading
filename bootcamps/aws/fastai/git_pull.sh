#!/bin/bash

# within fastai-eval-instance
mkdir -p tutorials/fastai/
cd ~/tutorials/fastai/
git clone https://github.com/fastai/course-v3
cd course-v3
git checkout .
git pull
sudo /opt/anaconda3/bin/conda update conda
sudo /opt/anaconda3/bin/conda install -c fastai fastai
conda update conda
conda install -c pytorch -c fastai fastai pytorch 

jupyter notebook