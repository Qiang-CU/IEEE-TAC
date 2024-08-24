import os
import time
import sys
import numpy as np
import random as rd
from mpi4py import MPI

from CSGD import CSGD
from DSGD import DSGD
from createData import CreateData
from quadProblem import QuadProblem
from communication import DecentralizedAggregation
from util import creat_mixing_matrix, create_sampling_time, compute_spectral_gap

def gen_data():
    save_dir = 's1_data/'
    data = CreateData(dim = 10, num_agent=20, num_local_data=500, save_dir=save_dir)
    data.generateData()


def ex_csgd():   
    """
        Run DSGD ant its variants algorithms
    """

    logMaxIter = 5
    batch = 5
    num_agent = 20
    save_dir = './s1_res-CSGD/'
    data_dir = 's1_data/'

    problem = QuadProblem(num_agent=num_agent, data_path=data_dir)
    csgd = CSGD(problem, num_agent, batch, logMaxIter, save_dir, lg_flag=True)
    csgd.fit()


def fixMixingMat():
    num_agent = 20
    graph = 'RingGraph'
    W = creat_mixing_matrix(num_agent, graph, self_weight=0.5)
    np.save('./s1_data/' + f'MixingMat-{graph}-NumAgent{num_agent}.npy', W)


def ex_dsgd():
    """
        Run DSGD ant its variants algorithms
    """

    rho_list = [0.7, 0.3]
    save_dir = './s1_res-DSGD/'
    data_dir = 's1_data/'
    graph = 'RingGraph'
    num_agent = 20
    num_trails = 2
    batch = 5


    problem = QuadProblem(num_agent=num_agent, data_path= data_dir)

    for rep in range(num_trails):
        for rho in rho_list:
            hybrid_dsgd = DSGD(problem, algo_type='hybrid', graph_type=graph, num_agent=num_agent, logMaxIter=5, save_dir=save_dir, data_dir=data_dir, batch=batch, rho = rho)
            hybrid_dsgd.fit(rep=rep)

        hete_dsgd = DSGD(problem, algo_type='hete', graph_type=graph, num_agent=num_agent, logMaxIter=5, save_dir=save_dir, data_dir=data_dir, batch=batch)
        hete_dsgd.fit(rep=rep)

        homo_dsgd = DSGD(problem, algo_type='homo', graph_type=graph, num_agent=num_agent, logMaxIter=5, save_dir=save_dir, data_dir=data_dir, batch=batch)
        homo_dsgd.fit(rep=rep)


if __name__ == '__main__':
    """ 
        To compare the CSGD and DSGD algorithms based on Homo, Partial Hete, Heterogeneous Data
        Parameters Settings: num_agent = 20
                            log_maxiter = 5
                            batch = 2
    """
    # Only need to run python main.py 
    # gen_data()
    # fixMixingMat()


    # Run Algorithms
    # ex_csgd()
    ex_dsgd()