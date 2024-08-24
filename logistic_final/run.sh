#!/bin/bash

# 生成数据
python createData.py

# get optimal solution
python main_getopt.py

# run Centralized SGD
mpirun -np 5 python CSGD.py

# Set mixing matrix
python DSGD_pre.py
 
# Run DSGD
mpirun -np 30 python DSGD.py