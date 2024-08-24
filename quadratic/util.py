import os
import numpy as np
import scipy.sparse
import networkx as nx

def create_sampling_time(logMaxIter, log_scale=True):
    """生成对数刻度或者正常刻度，sample_num记录metric运行的时间点"""
    maxIter = int(10**logMaxIter)
    num_points = int(min(1000, int(maxIter*0.2)))
    
    if log_scale:
        sample_num = np.geomspace(1, 10**logMaxIter, num_points, endpoint=False, dtype=int)
        return np.insert(np.unique(sample_num), 0, 0)
    else:
        sample_num = np.arange(0, maxIter, step=(maxIter)/num_points, dtype=int)  # 选取测算measurement的时间点
        return sample_num

def compute_spectral_gap(weighted_adj_matrix):
    eigenvalues = np.linalg.eigvals(weighted_adj_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    # 计算谱间隙（第二小的特征值与第一小的特征值之差）
    spectral_gap = sorted_eigenvalues[0] - sorted_eigenvalues[1]
    return spectral_gap

def creat_mixing_matrix(num_agent, graph_type, self_weight):
    if graph_type == 'RingGraph':
        A = np.zeros([num_agent, num_agent])  # adjacent matrix
        W = np.zeros([num_agent, num_agent])  # weighted matrix
        for i in range(num_agent):
            if i == 0:
                A[0, 1] = 1
                A[num_agent - 1, 0] = 1
                continue
            next_node = i + 1
            before_node = i - 1
            if next_node >= num_agent:
                next_node = 0
            if before_node < 0:
                before_node = num_agent - 1
            A[i, next_node] = 1
            A[next_node, i] = 1
            A[before_node, i] = 1
            A[i, before_node] = 1

        for i in range(num_agent):
            # Guarrante the sum of row is 1
            non_zero_num = len(np.nonzero(A[i, :])[0])
            weight = (1 - self_weight) / non_zero_num
            for j in range(num_agent):
                if i == j:
                    W[i][j] = self_weight
                elif A[i, j] != 0:
                    W[i, j] = weight
        spectral_gap = compute_spectral_gap(W)
        print(f'Ring Graph, spectral gap is: {spectral_gap}' )
        return W

    elif graph_type == 'FullyConnectedGraph':
        W = (np.ones((num_agent, num_agent)) - np.eye(num_agent)) * (1 - self_weight) / (num_agent - 1) + np.eye(
            num_agent) * self_weight
        spectral_gap = compute_spectral_gap(W)
        print(f'Fully Connected Graph, spectral gap is: {spectral_gap}' )
        return W

    elif graph_type == 'LineGraph':
        W = np.zeros((num_agent, num_agent))
        for i in range(num_agent):
            W[i, i] = self_weight
            if i == 0:
                W[i, i + 1] = 1 - self_weight
            elif i == num_agent - 1:
                W[i, i - 1] = 1 - self_weight
            else:
                W[i, i + 1] = (1 - self_weight) / 2
                W[i, i - 1] = (1 - self_weight) / 2
        return W

    elif graph_type == 'ER_graph':
        g = nx.erdos_renyi_graph(num_agent, p=np.log(n)/n, seed=123)
        while not nx.is_connected(g):
            g = nx.erdos_renyi_graph(num_agent, p=np.log(n)/n)
        adjacency_matrix = nx.adjacency_matrix(g).todense()
        degree = np.sum(adjacency_matrix, axis=0) #+ 1
        W = np.zeros((num_agent, num_agent)) #mixing matrix

        for i in range(num_agent):
            for j in range(num_agent):
                if adjacency_matrix[i, j] == 1 and i != j:
                    W[i, j] = min(1/degree[i], 1/degree[j])
        for i in range(num_agent):
            W[i, i] = 1 - np.sum(W[i, :])

        spectral_gap = compute_spectral_gap(W)
        print(f'ER Graph, spectral gap is: {spectral_gap}' )
        return W
    else:
        print('No Graph Name Matches!')



class plot_figure(object):
    def __init__(self, algo_name, dir, sub_sample=20, log_flag = False, metric = "dist2ps"):
        self.sub_sample = sub_sample
        self.dir = dir
        self.algo_name = algo_name
        self.log_flag = log_flag
        self.metric = metric

        self.num_trails, self.res, self.xvals = self.load_data(algo_name)
        self.z = 1.96/np.sqrt(self.num_trails) # 95% confidence， 1.645-90%


    def load_data(self, algo_name):
        file = [f for f in os.listdir(self.dir) if algo_name in f]
        # 读取这些文件的数据
        files = [np.load(os.path.join(self.dir, f), allow_pickle=True) for f in file]
        xvals = files[0].item().get('iter')

        num_points = int(1e3)
        log_indices = np.logspace(0, np.log10(len(xvals) - 1), num=num_points).astype(int)
        sample_iter = np.array(xvals)[log_indices]
        res = []
        for f in files:
            data = np.array(f.item().get(self.metric))
            res.append(data[log_indices])
        res = np.array(res)
        return len(file), res, sample_iter

    def plot_lines(self, ax, color, line='-', label='', plot_star = False, shadow_flag=True, legend=True):
        mean = np.mean(self.res, axis = 0)
        std = np.std(self.res, axis = 0)
        lb = np.squeeze(mean - self.z * std / np.sqrt(self.num_trails))
        ub = np.squeeze(mean + self.z * std / np.sqrt(self.num_trails))

        ax.plot( self.xvals[0::self.sub_sample], mean[0::self.sub_sample], label=label, color=color,linestyle=line, linewidth=2)
        if shadow_flag:
            ax.fill_between(self.xvals[0::self.sub_sample], lb[0::self.sub_sample], ub[0::self.sub_sample], color=color, alpha=.05)
        if legend:
            ax.legend(prop={'size': 12})
        if self.log_flag:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=13)

        if plot_star:
            ax.plot(self.xvals[0], mean[0], marker = '*', color = color, markerfacecolor=color,ms=15)
