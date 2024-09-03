import numpy as np
from mpi4py import MPI
import random


class QuadProblem(object):
    def __init__(self, num_agent, data_path):
        
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.num_agent = num_agent
        self.data_path = data_path
        self.local_data, self.all_data = self.load_data()

        self.num_local_data = len(self.local_data['Ai'])
        self.num_all_data = len(self.all_data['Ai'])

        self.dim = self.local_data['Ai'][0].shape[0]
        self.theta_opt = self.get_opt_sol()
    
    def load_data(self):
        all_data = {'Ai': None, 'bi': None}
        raw_data = np.load(self.data_path + f'TotalAgent{self.num_agent}-AllData.npy', allow_pickle=True).item()
        all_data['Ai'] = np.array(raw_data['Ai'])
        all_data['bi'] = np.array(raw_data['bi'])

        local_data = {'Ai': None, 'bi': None}
        raw_data = np.load(self.data_path + f'TotalAgent{self.num_agent}-agent{self.rank}.npy', allow_pickle=True).item()
        local_data['Ai'] = np.array(raw_data['Ai'])
        local_data['bi'] = np.array(raw_data['bi'])
 
        return local_data, all_data

    def get_opt_sol(self): # 小心不要算错了，否则会出现先下降后持平的问题
        Ai = np.sum([(x + x.T) for x in self.all_data['Ai']], axis=0)
        bi = - np.sum(self.all_data['bi'], axis=0)
        A_inv = np.linalg.inv(Ai)
        theta_star = np.dot(A_inv, bi)
        return theta_star

    def get_samples(self, batch, algo_type=None, ratio=None):
        """
        Get random vectors, agent_id<0 -> centralized algo, decen algo otherwise
        """

        if algo_type in ['cen', 'homo']:
            dataid = random.sample(range(0, self.num_all_data), batch)
            samples = {'Ai': self.all_data['Ai'][dataid], 'bi': self.all_data['bi'][dataid]}
            return samples
        elif algo_type == 'hybrid':
            samples = {'Ai': None, 'bi': None}
            if np.random.uniform(0,1) < ratio:
                idx = random.sample(range(0, self.num_all_data), batch)
                samples['Ai'] = self.all_data['Ai'][idx]
                samples['bi'] = self.all_data['bi'][idx]
            else:
                idx = random.sample(range(0, self.num_local_data), batch)
                samples['Ai'] = self.local_data['Ai'][idx]
                samples['bi'] = self.local_data['bi'][idx]
            return samples
        elif algo_type == 'hete':
            dataid = random.sample(range(0, self.num_local_data), batch)
            samples = {'Ai': self.local_data['Ai'][dataid], 'bi': self.local_data['bi'][dataid]}

            return samples
        else:
            raise 'Error, no sample scheme ! \n'


    def get_stoc_grd(self, Samples, theta):
        theta = theta.flatten()
        Ai = np.array([(x + x.transpose()) for x in Samples['Ai']])
        A_i = np.mean(Ai, axis=0)
        b_i = np.mean(Samples['bi'], axis=0)
        grd = A_i @ theta + b_i
        return grd
    