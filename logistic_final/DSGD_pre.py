import os
import time
import sys
import numpy as np
import random as rd
from mpi4py import MPI

from util import creat_mixing_matrix,compute_spectral_gap

# num_agent = 25
# graph = 'RingGraph'
# W = creat_mixing_matrix(num_agent, graph, self_weight=0.5)
# print('spectral gap of ring graph is ', compute_spectral_gap(W))

# # 保存mixing matrix矩阵到子文件夹
# np.save(os.path.join('data/', f'MixingMat-{graph}-NumAgent{num_agent}.npy'), W)


num_agent = 30
graph = 'RingGraph'
W = creat_mixing_matrix(num_agent, graph, self_weight=0.5)
print(f'spectral gap of  {graph} is ', compute_spectral_gap(W))

# 保存mixing matrix矩阵到子文件夹
np.save(os.path.join('data/', f'MixingMat-{graph}-NumAgent{num_agent}.npy'), W)