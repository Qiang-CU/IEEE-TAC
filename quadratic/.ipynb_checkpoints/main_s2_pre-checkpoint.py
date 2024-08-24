import os
import time
import sys
import numpy as np
import random as rd
from mpi4py import MPI

from util import creat_mixing_matrix,compute_spectral_gap
from createData import CreateData
from quadProblem import QuadProblem

num_agent_list = [10, 20, 30, 40, 50]
spectral_gap_list = []
graph = 'RingGraph'

for num_agent in num_agent_list:
    # 创建对应的子文件夹
    folder_path = f's2_data/NumAgent{num_agent}'
    save_path = f's2_res/NumAgent{num_agent}'
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    data = CreateData(dim = 10, num_agent=num_agent, num_local_data=300, save_dir=folder_path)
    data.generateData()

    W = creat_mixing_matrix(num_agent, graph, self_weight=0.5)
    spectral_gap_list.append(compute_spectral_gap(W))
    
    # 保存mixing matrix矩阵到子文件夹
    np.save(os.path.join(folder_path, f'MixingMat-{graph}-NumAgent{num_agent}.npy'), W)

print(np.sqrt(num_agent_list)/spectral_gap_list)


"Run Command: python main_s2_prep.py"