#! /usr/bin/env bash

python3.11 -m venv venv

source source.bash

pip install --upgrade pip

pip install open3d numpy scikit-learn plotly black
