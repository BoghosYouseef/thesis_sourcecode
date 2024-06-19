#!/bin/bash
#backup patch models data
. backup.bash
git pull
. clear_all_generated_data.bash
cd ../

module load lang/Python
module load numlib/cuDNN

python3 -m venv .venv
pip install --upgrade pip
cp ../requirements.txt .
pip install -r requirements.txt



sbatch scripts/hpc_scripts/script1.bash
