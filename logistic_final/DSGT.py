import os
import time
import sys
import numpy as np
import random as rd
from mpi4py import MPI

from logistic import LogisticMin
from communication import DecentralizedAggregation
from util import creat_mixing_matrix, create_sampling_time, compute_spectral_gap


class DSGT(object):
    
    def __init__(self, problem, algo_type, graph_type, num_agent, logMaxIter, data_dir, save_dir, log_scale=False, batch=5):
        self.problem = problem
        self.dim = problem.dim
        self.sample_time = create_sampling_time(logMaxIter, log_scale = log_scale)
        self.batch = batch
        self.num_agent = num_agent
        self.graph = graph_type
        self.algo_type = algo_type
        self.maxIter = int(10**logMaxIter)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.root_rank = 0 # 这个process负责收集数据，写数据
        
        # MPI setting
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.W = self.getW()
        neighbour_dict = self.neighbour_weights()

        self.communicator = DecentralizedAggregation(neighbour_dict)
        
        # 
        self.theta = np.ones(self.dim + 1, ) * np.random.uniform(-0.1, 0.1)
        self.tracking_y = np.zeros(self.dim + 1, )   # tracking variabel y
        self.last_grd = np.zeros(self.dim + 1, )   # last gradient

        self.metric = {'iter': [], 'mse': [], 'wmse': []}

        self.theta_opt = self.load_optsol()
    
    def getW(self):
        file_name = self.data_dir + f'MixingMat-{self.graph}-NumAgent{self.num_agent}.npy'
        if os.path.exists(file_name):
            W = np.load(file_name)
            return W
        else:
            raise FileNotFoundError(f"File '{file_name}' does not exist.")
        return W
    
    # DSGT does not need plain communication
    # def communication(self):
    #     prm = np.copy(self.theta)
    #     self.theta = self.communicator.agg(prm, op="weighted_avg")

    def load_optsol(self):
        file_name = f'{self.data_dir}/theta_opt.npy'
        raw_data = np.load(file_name, allow_pickle=True)
        theta_opt = np.array(raw_data)
        return theta_opt
    
    def neighbour_weights(self):
        """根据图的权值矩阵W返回节点rank的邻居节点"""
        neighbours = np.where(self.W[self.rank] > 0)[0]
        neighbour_w = {i: self.W[self.rank][i] for i in neighbours}
        return neighbour_w
    
    def record(self, t, collect_theta):
        avg_theta = np.mean(collect_theta, axis=0)
        distances_squared = np.linalg.norm(collect_theta - self.theta_opt, axis=1) ** 2
        wmse = np.max(distances_squared)
        mse = np.linalg.norm(avg_theta - self.theta_opt, ord=2) ** 2
        self.update_metric(t, mse, wmse)
    
    def update_metric(self, t, mse, wmse):
        self.metric['iter'].append(t)
        self.metric['mse'].append(mse)
        self.metric['wmse'].append(wmse)
    
    def stepsize(self, t):
        a0 = 5 #50 #1e2
        a1 = 100 #1e4
        return a0 / (a1 + t)
        
    def save(self, rep):        
        if self.algo_type == 'homo':
            np.save(self.save_dir + f"rep{rep}-gt-homo-{self.graph}.npy", self.metric)
        elif self.algo_type == 'hete':
            np.save(self.save_dir + f"rep{rep}-gt-hete-{self.graph}.npy", self.metric)
        else:
            raise ValueError('no matching distributed algorithms.')

    def sample(self):

        if self.algo_type == 'homo':
            idx = np.random.choice(self.problem.total_num_sample, self.batch, replace=False)
            return self.problem.allfeature[idx], self.problem.alllabel[idx]

        elif self.algo_type == 'hete':

            idx = np.random.choice(self.problem.local_num_sample, self.batch, replace=False)
            return self.problem.local_feature[idx], self.problem.local_label[idx]
    
    def fit(self, rep):
        
        for t in range(self.maxIter):

            temp = self.theta - self.stepsize(t) * self.tracking_y
            self.theta = self.communicator.agg(temp, op = 'weighted_avg')
            X, Y = self.sample()
            new_grd = self.problem.stoc_grad(X, Y, self.theta)
            temp2 = self.communicator.agg(self.tracking_y, op='weighted_avg')
            self.tracking_y = temp2 + new_grd - self.last_grd

            self.last_grd = np.copy(new_grd) #save current grad for the next iter
            
            
            if t in self.sample_time:
                collect_theta = self.comm.gather(self.theta, root=self.root_rank) 
                if self.rank == self.root_rank:
                    self.record(t, collect_theta)
            
            if self.rank == 0 and t % 100 == 0:
                print(f'Gradient Tracking-{self.algo_type}: Iter Percentage {t/self.maxIter:.0%}, MSE: {self.metric["mse"][-1]:.3e}, Worst MSE: {self.metric["wmse"][-1]:.3e}')
                sys.stdout.flush()  # 强制刷新缓冲区
                
        if self.rank == self.root_rank:
            self.save(rep)



if __name__ == "__main__":
    """
        mpirun -np 30 python DSGT.py
        mpiexec --allow-run-as-root -np 12 python DSGT.py
    """
    dir = 'res/'
    graph = 'RingGraph' #'FullyConnectedGraph' #'RingGraph'
    num_agent = 30
    num_trails = 3

    problem = LogisticMin(num_agent=num_agent, data_dir='data/')


    for rep in range(num_trails):

        hete_gt = DSGT(problem, algo_type='hete', graph_type=graph, num_agent=num_agent, logMaxIter=5, save_dir=dir, batch=4, data_dir = 'data/', log_scale=True)
        hete_gt.fit(rep=rep)

        homo_gt = DSGT(problem, algo_type='homo', graph_type=graph, num_agent=num_agent, logMaxIter=5, save_dir=dir, batch=4, data_dir = 'data/', log_scale=True)
        homo_gt.fit(rep=rep)
