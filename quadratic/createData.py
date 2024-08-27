import numpy as np
import os 
import random as rd


def is_psd(A):
    eigenvalues, _ = np.linalg.eig(A)
    return np.all(eigenvalues >= 0)


class CreateData(object):

    def __init__(self, dim, num_agent, num_local_data, save_dir):
        self.dim = dim
        self.num_agent = num_agent
        self.num_local_data = num_local_data
        self.num_data = self.num_local_data * self.num_agent #整个数据集中data points的个数

        self.Ai = []
        self.bi = []
        self.save_dir = save_dir
    
    def generateData(self):

        # generate random matrices, st their mean is PD matrix
        
        #TODO: the following data generation method may not propery for decentralized algo, see logistic simulation example.
        for _ in range(self.num_agent):
            Ai = []
            bi = []

            # s1: generate a random symmetric matrix
            a = np.random.randn(self.dim, self.dim)
            true_b = np.random.randn(self.dim,) 
            a = (a + a.T) / 2
            cov_mat = np.dot(a, a.T)
            epsilon = 1e-3 # make sure that cov_mat is pd
            true_a = cov_mat + epsilon * np.eye(self.dim) #now true_a must be pd matrix
            
            # step 2: replace special diag elements to Normal(0,2)
            diag_vec = np.copy(np.diagonal(true_a))
            num_special_index = 2
            special_index = rd.sample(range(self.dim), num_special_index) 
            diag_vec[special_index] = np.random.normal(loc = 0, scale = 2)
            np.fill_diagonal(true_a, diag_vec)
            
            for _ in range(self.num_local_data):

                tempbi = true_b + np.random.randn(self.dim,) 
                tempAi = true_a + np.random.randn(self.dim, self.dim) 
                
                Ai.append(tempAi)
                bi.append(tempbi)
                
            self.Ai.append(Ai)
            self.bi.append(bi)

        self.saveData()


    def saveData(self):
        
        for i in range(self.num_agent):
            data = {'Ai': self.Ai[i], 'bi': self.bi[i]}
            np.save(os.path.join(self.save_dir, f'TotalAgent{self.num_agent}-agent{i}.npy'), data)
        
        tempAi = np.array([ A for sublist in self.Ai for A in sublist])
        tempbi = np.array( [ b for sublist in self.bi for b in sublist] )
        # Note: do not shuffle tempAi, tempbi separately, because they come from different true Ai and bi
        combined = list(zip(tempAi, tempbi))
        np.random.shuffle(combined)
        tempAi, tempbi = zip(*combined)
        tempAi = np.array(tempAi)
        tempbi = np.array(tempbi)

        
        alldata = {'Ai': tempAi, 'bi': tempbi}
        np.save(self.save_dir + f'/TotalAgent{self.num_agent}-AllData.npy', alldata)


if __name__ == "__main__":
    save_dir = 's1_data/'
    data = CreateData(dim = 10, num_agent=20, num_local_data=500, save_dir=save_dir)
    data.generateData()