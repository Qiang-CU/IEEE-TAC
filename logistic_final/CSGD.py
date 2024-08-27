import numpy as np
import sys
from mpi4py import MPI
from tqdm import tqdm
from util import create_sampling_time


class CSGD(object):

    def __init__(self, problem, num_agent, batch, logMaxIter, a0, a1, save_dir, data_dir = 'data', log_scale=False):
        self.problem = problem
        self.dim = problem.dim
        self.maxIter = int(10**logMaxIter)
        self.batch = batch
        self.num_agent = num_agent
        self.save_dir = save_dir
        self.data_dir = data_dir

        # MPI setting
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.metric = {'iter': [], 'mse': [], 'wmse': []}
        self.prm = np.ones(self.dim + 1,) * 0.2

        self.a0 = a0
        self.a1 = a1

        self.sample_time = create_sampling_time(logMaxIter, log_scale=log_scale)
        self.theta_opt = self.load_optsol()
    
    def stepsize(self, t):
        # a0 = 1 / self.problem.lam
        # a1 = 8*self.problem.lipschitz**2 / self.problem.lam
        
        a0 = self.a0
        a1 = self.a1
        
        return a0 / (a1 + t)

    def load_optsol(self):
        file_name = f'{self.data_dir}/theta_opt.npy'
        raw_data = np.load(file_name, allow_pickle=True)
        theta_opt = np.array(raw_data)
        return theta_opt

    def sample(self):
        idx = np.random.choice(self.problem.total_num_sample, self.batch * self.num_agent, replace=False)
        return self.problem.allfeature[idx], self.problem.alllabel[idx]


    def record(self, t):
        self.metric['iter'].append(t)
        mse = np.linalg.norm(self.prm - self.theta_opt, ord=2)**2
        self.metric['mse'].append(mse)
        self.metric['wmse'].append(mse) # 由于centralized algo没有worst case, 所以直接用mse代替
    
    def save(self):
        np.save(self.save_dir + f"rep{self.rank}-csgd.npy", self.metric)
    

    def fit(self):
        for t in range(self.maxIter):
            X, Y = self.sample()
            self.prm = self.prm - self.stepsize(t) * self.problem.stoc_grad(X, Y, self.prm)
            if t in self.sample_time: self.record(t) 
            if t % 100 == 0 and self.rank == 0:
                print(f'Centralized SGD: Iter Percentage: {t/self.maxIter:.0%}, MeanSquareError: {self.metric["mse"][-1]:.3e}')
                sys.stdout.flush()  # 强制刷新缓冲区
                
        self.save()


if __name__ == "__main__":
    """
        mpirun -np num_trails python CSGD.py
        mpirun -np 5 python CSGD.py
        mpiexec --allow-run-as-root -np 5 python CSGD.py
    """
    from logistic import LogisticMin
    a0, a1 = [5, 50] # [50, 1e4] #[10*1e3, 1e4], [5*1e3, 1e4] [1e3, 1e4]  #[10, 200]

    data_dir = 'data/'
    batch = 4
    save_dir = 'res/'
    problem = LogisticMin(num_agent=30, data_dir = data_dir)
    csgd = CSGD(problem, num_agent=30, batch=batch, logMaxIter=5, a0=a0, a1=a1, save_dir=save_dir, log_scale=True)
    csgd.fit()