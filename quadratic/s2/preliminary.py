import numpy as np
import os

from createData import CreateData
from util import creat_mixing_matrix, is_doubly_stochastic, compute_spectral_gap



save_dir = 's2_data/'
num_agent_list = list(range(5, 55, 5))
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
    if not is_doubly_stochastic(W):
        raise ValueError("W is not doubly stochastic")

    spectral_gap = compute_spectral_gap(W)
    if spectral_gap.imag != 0:
        ValueError("Spectral gap is not real")
    else:
        spectral_gap = spectral_gap.real

    print(f'{num_agent}-agent {graph}, spectral gap is: {spectral_gap}' )

    filename = f's2_mixMat/MixingMat-{graph}-NumAgent{num_agent}.npy'
    np.save(filename, W)