import os
import time
import sys
import numpy as np
import random as rd
from mpi4py import MPI

from quadProblem import QuadProblem
from communication import DecentralizedAggregation
from util import creat_mixing_matrix, create_sampling_time, compute_spectral_gap

class DSGD(object):
    
    def __init__(self, problem, algo_type, graph_type, num_agent, logMaxIter, data_dir, save_dir, rho=0, lg_flag=True, batch=1, a0=10, a1=500):
        self.problem = problem
        self.dim = problem.dim
        self.sample_time = create_sampling_time(logMaxIter, log_scale=lg_flag)
        self.algo_type = algo_type
        self.batch = batch
        self.num_agent = num_agent
        self.graph = graph_type
        self.maxIter = int(10**logMaxIter)
        self.root_rank = 0 # 这个process负责收集数据，写数据

        self.data_dir = data_dir
        self.save_dir = save_dir
        
        # MPI setting
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # 
        self.theta = np.random.rand(self.dim, )
        self.rho = rho # rho * D_i + rho * D
        self.W = self.getW()

        #
        self.a0 = a0
        self.a1 = a1

        neighbour_dict = self.neighbour_weights()
        self.communicator = DecentralizedAggregation(neighbour_dict)
        self.metric = {'iter': [], 'mse': [], 'wmse': []}
    
    def getW(self):
        file_name = self.data_dir
        W = np.array(np.load(file_name, allow_pickle=True))
        return W
        # else:
        #     # 路径下没有mixing matrix文件,那就创建一个
        #     W = creat_mixing_matrix(num_agent, graph, self_weight=0.3)
        #     np.save('./s1_data/' + f'MixingMat-{graph}-NumAgent{num_agent}.npy', W)
        #     print(f"File '{file_name}' does not exist, but created now")
        #     return W
    
    def update_metric(self, t, mse, wmse):
        self.metric['iter'].append(t)
        self.metric['mse'].append(mse)
        self.metric['wmse'].append(wmse)
    
    def communication(self):
        prm = np.copy(self.theta)
        self.theta = self.communicator.agg(prm, op="weighted_avg")
    
    def neighbour_weights(self):
        """根据图的权值矩阵W返回节点rank的邻居节点"""
        neighbours = np.where(self.W[self.rank] > 0)[0]
        neighbour_w = {i: self.W[self.rank][i] for i in neighbours}
        return neighbour_w
    
    def record(self, t, collect_theta):
        avg_theta = np.mean(collect_theta, axis=0)
        
        distances_squared = np.linalg.norm(collect_theta - self.problem.theta_opt, axis=1) ** 2
        wmse = np.max(distances_squared)
        mse = np.linalg.norm(avg_theta - self.problem.theta_opt, ord=2) ** 2
        self.update_metric(t, mse, wmse)
    
    def stepsize(self, t):
        return self.a0 / (self.a1 + t)
        
    def save(self, rep):        
        if self.algo_type == 'homo':
            np.save(self.save_dir + f"rep{rep}-dsgd-homo-{self.graph}.npy", self.metric)
        elif self.algo_type == 'hybrid':
            np.save(self.save_dir + f"rep{rep}-dsgd-hybrid-{self.graph}-rho{self.rho}.npy", self.metric)
        elif self.algo_type == 'hete':
            np.save(self.save_dir + f"rep{rep}-dsgd-hete-{self.graph}.npy", self.metric)
        else:
            raise ValueError('no matching distributed algorithms.')
    
    def fit(self, rep):

        for t in range(self.maxIter):
            self.communication()
            Samples = self.problem.get_samples(batch=self.batch, algo_type=self.algo_type, ratio=self.rho)
            self.theta = self.theta - self.stepsize(t) * self.problem.get_stoc_grd(Samples, self.theta)
    
            if t in self.sample_time:
                collect_theta = self.comm.gather(self.theta, root=self.root_rank) 
                if self.rank == self.root_rank: self.record(t, collect_theta)

            if t % 100 == 0 and self.rank == 0:
                print(f'Decentralized SGD-{self.algo_type}: Iter Percentage {t/self.maxIter:.0%}, MSE: {self.metric["mse"][-1]:.3e}, Worst MSE: {self.metric["wmse"][-1]:.3e}')
                sys.stdout.flush()  # 强制刷新缓冲区

        if self.rank == self.root_rank:
            self.save(rep)

