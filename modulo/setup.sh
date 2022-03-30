#! /bin/bash

echo "CREATING VIRTUAL ENV"
python -m venv venv
source ./venv/bin/activate

echo "INSTALLING DEPENDENCIES"
pip install -r requirements.txt

echo "CREATING FOLDER STRUCTURE"
mkdir sessions
