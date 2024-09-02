import sys
import numpy as np
from mpi4py import MPI
from util import create_sampling_time


class CSGD(object):

    def __init__(self, problem, num_agent, batch, logMaxIter, save_dir, lg_flag=True, a0=10, a1=500):
        self.problem = problem
        self.dim = problem.dim
        self.num_agent = num_agent
        self.batch = num_agent * batch

        self.logMaxIter = logMaxIter
        self.maxIter = int(10**logMaxIter)
        self.sample_time = create_sampling_time(logMaxIter, log_scale=lg_flag)
        self.dir = save_dir

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.a0 = a0
        self.a1 = a1

        self.metric = {'iter': [], 'mse': [], 'wmse': []}
        self.theta = np.random.rand(self.dim, )

    def stepsize(self, t):
        return self.a0 / (t + self.a1)
    
    def record(self, t):
        self.metric['iter'].append(t)
        mse = np.linalg.norm(self.theta - self.problem.theta_opt, ord=2)**2
        self.metric['mse'].append(mse)
        self.metric['wmse'].append(mse) # use mse as placeholder, will use in the plot wmse figure.

    def save(self):
        print(f'rank{self.rank} save result to {self.dir}')
        np.save(self.dir + f"rep{self.rank}-csgd.npy", self.metric)

    def fit(self):

        for t in range(self.maxIter):
            Samples = self.problem.get_samples(batch = self.batch , algo_type='cen')
            grad = self.problem.get_stoc_grd(Samples, self.theta)
            self.theta = self.theta - self.stepsize(t) * grad

            if t in self.sample_time: self.record(t)
            if t % 100 == 0 and self.rank == 0:
                print(f'Centralized SGD: Iter Percentage: {t/self.maxIter:.0%}, MeanSquareError: {self.metric["mse"][-1]:.3e}')
                sys.stdout.flush()  # 强制刷新缓冲区        
        self.save()
        

if __name__ == "__main__":
    """
    env4 有mpi
        mpiexec -np 20 python csgd.py
    """
    from quadProblem import QuadProblem
    
    logMaxIter = 4
    batch = 1
    num_agent = 20
    save_dir = './s1_res/'
    data_dir = './s1_data/'

    problem = QuadProblem(num_agent=num_agent, data_path = data_dir)
    csgd = CSGD(problem, num_agent, batch, logMaxIter, save_dir, lg_flag=True)
    csgd.fit()