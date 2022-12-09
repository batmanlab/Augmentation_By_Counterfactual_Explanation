#!/bin/bash
# use the bash shell

source /jet/home/nmurali/asc170022p/nmurali/anaconda3/etc/profile.d/conda.sh 
conda activate pl

eval "$*" # The argument passed to this script should be the command to launch a python script