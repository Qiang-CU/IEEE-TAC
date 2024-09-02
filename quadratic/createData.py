import numpy as np
import os 
import random as rd

def compute_spectral_gap(weighted_adj_matrix):
    eigenvalues = np.linalg.eigvals(weighted_adj_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    # 计算谱间隙（第二小的特征值与第一小的特征值之差）
    spectral_gap = sorted_eigenvalues[0] - sorted_eigenvalues[1]
    return spectral_gap


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


def create_mixing_matrix_star_graph(num_agent):
    adj_matrix = np.zeros((num_agent, num_agent), dtype=int)
    for i in range(1, num_agent):
        adj_matrix[0, i] = 1
        adj_matrix[i, 0] = 1

    W = np.zeros((num_agent, num_agent)) #mixing matrix
    degree = np.sum(adj_matrix, axis=0) + 1
    for i in range(num_agent):
        for j in range(num_agent):
            if adj_matrix[i, j] == 1 and i != j:
                W[i, j] = min(1/degree[i], 1/degree[j])
    for i in range(num_agent):
        W[i, i] = 1 - np.sum(W[i, :])
    return W


def is_doubly_stochastic(matrix):
    matrix = np.array(matrix)
    if np.any(matrix < 0):
        return False
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    return np.allclose(row_sums, 1) and np.allclose(col_sums, 1)


if __name__ == "__main__":
    save_dir = 's2_data/'
    num_agent_list = list(range(5, 55, 5))

    for num_agent in num_agent_list:
        # 创建对应的子文件夹
        folder_path = f's2_data/NumAgent{num_agent}'
        save_path = f's2_res/NumAgent{num_agent}'
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        data = CreateData(dim = 10, num_agent=num_agent, num_local_data=300, save_dir=folder_path)
        data.generateData()

        W = create_mixing_matrix_star_graph(num_agent)
        if not is_doubly_stochastic(W):
            raise ValueError("W is not doubly stochastic")

        spectral_gap = compute_spectral_gap(W)
        if spectral_gap.imag != 0:
            ValueError("Spectral gap is not real")
        else:
            spectral_gap = spectral_gap.real

        print(f'{num_agent}-agent Star Graph, spectral gap is: {spectral_gap}' )

        filename = f's2_mixMat/MixingMat-StarGraph-NumAgent{num_agent}.npy'
        np.save(filename, W)