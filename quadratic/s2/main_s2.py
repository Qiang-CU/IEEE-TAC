import os
import time
import sys
import numpy as np
import random as rd
from mpi4py import MPI

from CSGD import CSGD
from DSGD import DSGD
from communication import DecentralizedAggregation
from util import creat_mixing_matrix, create_sampling_time, compute_spectral_gap
from createData import CreateData
from quadProblem import QuadProblem


num_agent = int(sys.argv[1]) # Note: input's type is str, you need to change the type
logMaxIter = 2
batch = 1
save_dir = f's2_res/NumAgent{num_agent}/'
data_dir = f's2_data/NumAgent{num_agent}/'
graph = 'RingGraph' #'StarGraph'
mixmat_dir = f's2_mixMat/MixingMat-{graph}-NumAgent{num_agent}.npy'
num_trails = 1

problem = QuadProblem(num_agent=num_agent, data_path=data_dir)


csgd = CSGD(problem, num_agent, batch, logMaxIter, save_dir, lg_flag=False, a0=10, a1=500)
csgd.fit()

for rep in range(num_trails):
    homo_dsgd = DSGD(problem, algo_type='homo', graph_type=graph, num_agent=num_agent, logMaxIter=logMaxIter, save_dir=save_dir,
                     data_dir=mixmat_dir, batch=batch, lg_flag=False, a0=10, a1 = 500)
    homo_dsgd.fit(rep=rep)

    if num_agent <= 20:
        hete_dsgd = DSGD(problem, algo_type='hete', graph_type=graph, num_agent=num_agent, logMaxIter=logMaxIter, save_dir=save_dir,
                        data_dir=mixmat_dir, batch=batch, lg_flag=False, a0=10, a1=5000)
        hete_dsgd.fit(rep=rep)




