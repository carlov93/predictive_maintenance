#!/bin/bash
# create virtual environment
pip3 install virtualenv
virtualenv venvMasterarbeit
source venvMasterarbeit/bin/activate
pip install ipykernel
ipython kernel install --user --name=masterarbeit

# install packages
pip3 install torch
pip3 install pandas
pip3 install tensorflow
pip3 install keras
pip3 install sklearn
pip3 install matplotlib