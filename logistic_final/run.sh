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
mpirun -np 30 python DSGD.py # er graph and ring graph

mpirun -np 30 python DSGT.py

mpirun -np 30 python EDAS.py

mpirun -np 30 python ExactDiff.py
