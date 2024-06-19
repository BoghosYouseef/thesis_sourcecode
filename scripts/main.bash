#!/bin/bash

cd ../

python3 -m venv .venv
pip install --upgrade pip
cp ../requirements.txt .
pip install -r requirements.txt

sbatch scripts/hpc_scripts/script1.bash
