#!/bin/bash


python opt_prm.py

python main.py

mpiexec --allow-run-as-root -np 5 python main.py

mpiexec -np 5 python main.py