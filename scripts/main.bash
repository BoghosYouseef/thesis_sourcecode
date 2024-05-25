#!/bin/bash

cd ../

py -m venv .venv
pip install -r requirements.txt


py ./src/main.py