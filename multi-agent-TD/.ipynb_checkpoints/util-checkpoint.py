import numpy as np 
import os

def create_sampling_time(logMaxIter, log_scale=True):
    """生成对数刻度或者正常刻度，sample_num记录metric运行的时间点"""
    num_points = int(2000)
    maxIter = int(10**logMaxIter)

    if log_scale:
        sample_num = np.geomspace(1, 10**logMaxIter, num_points, endpoint=False, dtype=int)
    else:
        sample_num = np.arange(1, maxIter, step=(maxIter)/num_points, dtype=int)  # 选取测算measurement的时间点
    
    sample_num = np.unique(sample_num).tolist()

    return sample_num

class plot_figure(object):
    def __init__(self, algo_name, dir, sub_sample=1, log_flag_ = False, plot_steadystate = False, plot_ps = False, metric='dist2opt'):
        
        self.import_package()
        
        self.sub_sample = sub_sample
        self.dir = dir
        self.algo_name = algo_name
        self.log_flag = log_flag_
        self.steady_state = plot_steadystate
        self.plot_ps = plot_ps
        self.metric = metric

        self.num_trails, self.res, self.xvals = self.load_data(algo_name)
        self.z = 1.96/np.sqrt(self.num_trails) # 95% confidence， 1.645-90%
    
    def import_package(self):
        import matplotlib
        import matplotlib.pyplot as plt

        # matplotlib.rcParams['ps.useafm'] = True
        # matplotlib.rcParams['pdf.use14corefonts'] = True
        # matplotlib.rcParams['text.usetex'] = True
        

    def load_data(self, algo_name):
        """从指定的数据文件中加载plot数据"""

        file = [f for f in os.listdir(self.dir) if algo_name in f]
        # 读取这些文件的数据
        files = [np.load(os.path.join(self.dir, f), allow_pickle=True) for f in file]
        # 读取measurement, i.e., consensus error, mean square error
        res = np.array([f.item().get(f"{self.metric}") for f in files])
        # 读取数据横坐标
        xvals = files[0].item().get('iter')
        self.ps = files[0].item().get('ps')

        if self.steady_state:
            self.bias = np.mean(np.array([f.item().get("bias") for f in files]))
        if self.plot_ps:
            self.ps = files[0].item().get('ps')
        
        message = f"""
        检查到有{len(res)}个数据文件: {file[:]} ...
        每个文件中gap list的长度有: {[len(r) for r in res]}
        """
        if self.log_flag:
            print(message)

        return len(file), res, xvals

    def plot_lines(self, ax, color, line='-', label='', plot_star = False):
        mean = np.mean(self.res, axis = 0)
        std = np.std(self.res, axis = 0)

        lb = np.squeeze(mean - self.z * std / np.sqrt(self.num_trails))
        ub = np.squeeze(mean + self.z * std / np.sqrt(self.num_trails))

        ax.plot( self.xvals[0::self.sub_sample], mean[0::self.sub_sample], label=label, color=color, lw=1.5, linestyle = '-')
        if self.plot_ps:
            ax.plot( self.xvals[0::self.sub_sample], self.ps * np.ones_like(self.xvals[0::self.sub_sample]), label=label, color='red', lw=1.5, linestyle = '-.')
        ax.fill_between(self.xvals, lb, ub, color=color, alpha=.05)

        if plot_star:
            ax.plot(self.xvals[0], mean[0], marker = '*', color = color, markerfacecolor=color,ms=15)
        
        if self.steady_state:
            ax.plot( self.xvals[0::self.sub_sample], self.bias * np.ones_like(self.xvals[0::self.sub_sample]), label=label, color=color, lw=1.2, linestyle = '--')


def creat_graph(num_agent, graph_type, self_weight):
    # This function can create RingGraph, FullyConnectedGraph, Line Graph
    # It will return weighted matrix W

    if graph_type == 'RingGraph':
        # Ring Graph Parameters
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

        # print(A)

        for i in range(num_agent):
            # Guarrante the sum of row is 1
            non_zero_num = len(np.nonzero(A[i, :])[0])
            weight = (1 - self_weight) / non_zero_num

            for j in range(num_agent):
                if i == j:
                    W[i][j] = self_weight
                elif A[i, j] != 0:
                    W[i, j] = weight
        # print(W)
        return W

    elif graph_type == 'FullyConnectedGraph':
        W = (np.ones((num_agent, num_agent)) - np.eye(num_agent)) * (1 - self_weight) / (num_agent - 1) + np.eye(
            num_agent) * self_weight

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
    else:
        print('No Graph Name Matches!')