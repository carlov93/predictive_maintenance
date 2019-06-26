#!/bin/bash
# create virtual environment
python -m venv venvMasterarbeit
source venvMasterarbeit/bin/activate
pip install ipykernel
ipython kernel install --user --name=masterarbeit

# install packages
pip install torch
pip install pandas
pip install tensorflow
pip install keras
pip install torchsummary
pip install sklearn
pip install matplotlib