import numpy as np
from createData import CreateData
from mpi4py import MPI
import os


class LogisticMin(object):

    def __init__(self, num_agent, data_dir):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.num_agent = num_agent
        self.data_dir = data_dir
        
        self.local_feature, self.local_label, self.allfeature, self.alllabel = self.load_data()
        

        # self.theta_opt =  np.load(f'./{self.data_dir}/theta_opt.npy', allow_pickle=True)
        # if algo_type == 'hete':
        #     self.mm = self.feature.shape[0] #
        #     self.m = self.feature.shape[0] * num_agent
        #     self.dim = self.feature.shape[1]
        # else:
        #     self.m, self.dim = self.feature.shape
        #     self.mm = np.copy(self.m)
        self.total_num_sample = self.allfeature.shape[0]
        self.dim = self.allfeature.shape[1] - 1
        self.local_num_sample, _ = self.local_feature.shape
        
        self.lam = 1  #0.01 / self.mm # regularization parameter
        # self.mu = self.lam + np.sum(np.square(np.linalg.norm(self.feature, axis=1))) / (4.0 * self.mm)
        
        # self.lipschitz = np.sum(np.square(np.linalg.norm(self.feature, axis=1)) / (4.0 * self.mm)) + self.lam

        # self.display()
        

    # def sample(self, batch):
    #     idx = np.random.choice(self.mm, batch, replace=False)
    #     return self.feature[idx], self.label[idx]


    def load_data(self):
        file_name = f'{self.data_dir}/TotalAgent{self.num_agent}-agent{self.rank}.npy'
        local_data = np.load(file_name, allow_pickle=True).item()
        local_feature = np.array(local_data['feature'])
        local_label = np.array(local_data['label'])

        file_name = f'{self.data_dir}/TotalAgent{self.num_agent}-AllData.npy'
        alldata = np.load(file_name, allow_pickle=True).item()
        allfeature = np.array(alldata['feature'])
        alllabel = np.array(alldata['label'])

        # add bias term
        local_feature = np.append(local_feature, np.ones((local_feature.shape[0], 1)), axis=1)
        allfeature = np.append(allfeature, np.ones((allfeature.shape[0], 1)), axis=1)

        return local_feature, local_label, allfeature, alllabel

    def stoc_grad(self, X, Y, theta):
        """
        stochastic gradient using tanh() function to approximate sigmoid()
        """
        if isinstance(Y, int): #对Y是整数的情况进行处理
            Y = np.array([Y])
        lin_comb = X @ theta
        tanh_output = np.tanh(lin_comb * 0.5)
        factor = ((tanh_output + 1) * 0.5 - Y)
        result = X * factor[:, np.newaxis] # 使用广播机制进行逐元素乘法
        result_batch_mean = np.mean(result, axis=0) + self.lam * np.append(theta[:-1], 0)
        # result_batch_mean = np.mean(result, axis=0) + self.lam * theta #no bias term
        
        return result_batch_mean
    
    def loss(self, X, Y, theta):
        n = X.shape[0]
        # t1 = 1.0/n * np.sum(-1.0 * np.multiply(Y, X @ theta) + np.log(1 + np.exp(X @ theta)))
        # t2 = self.lam / 2.0 * np.linalg.norm(theta)**2

        t1 = 1.0/n * np.sum(-1.0 * np.multiply(Y, X @ theta) + np.log(1 + np.exp(X @ theta)))
        t2 = self.lam / 2.0 * np.linalg.norm(theta[:-1])**2 #注意这里不考虑bias
        
        loss = t1 + t2
        return loss
    
    def grad(self, X, Y, theta):
        """
        full gradient
        """

        batch = X.shape[0]
        exp_tx = np.exp(X @ theta)
        c = exp_tx / (1 + exp_tx) - Y
        gradient = 1.0/batch * np.sum(X * c[:, np.newaxis], axis=0)  + self.lam * np.append(theta[:-1], 0)
        # gradient = 1.0/num_samples * np.sum(X * c[:, np.newaxis], axis=0) + self.lam * theta
        
        return gradient