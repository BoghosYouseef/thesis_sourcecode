#!/bin/bash
. backup.bash
git pull
. clear_all_generated_data.bash
cd ../
#backup patch models data

python3 -m venv .venv
pip install --upgrade pip
cp ../requirements.txt .
pip install -r requirements.txt
. clear
sbatch scripts/hpc_scripts/script1.bash
