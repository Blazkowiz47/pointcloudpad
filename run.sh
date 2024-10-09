#!/bin/bash


echo "Running Scripts"

conda init
conda activate point 
python custom.py
conda deactivate

conda activate base
python eval.py
conda deactivate

