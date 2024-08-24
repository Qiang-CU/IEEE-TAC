import numpy as np
import os

class CreateData(object):

    def __init__(self, dim, num_agent, num_local_data, save_dir):
        self.dim = dim
        self.num_agent = num_agent
        self.num_local_data = num_local_data
        self.save_dir = save_dir
        self.num_data = self.num_local_data * self.num_agent #整个数据集中data points的个数
        self.theta_true = self.generate_theta_ture() #每个agent的真实theta
        self.bias = np.random.uniform(0, 1, self.num_agent)

    def generate_theta_ture(self):
        theta_true = []
        l = np.linspace(-10, 10, self.num_agent+1) 
        for c_r in range(self.num_agent):
            theta_true.append(np.random.uniform(l[c_r], l[c_r+1], self.dim))
        return theta_true

    def generate_data(self):

        all_feature = []
        all_label = []
        for i in range(self.num_agent):
            feature = np.random.uniform(-10, 10, (self.num_local_data, self.dim))
            label = (np.sign(np.dot(feature, self.theta_true[i])+self.bias[i]) + 1)/2
            label = [int(x) for x in label]
            all_feature.append(feature)
            all_label.append(label)
            self.save_data(i, feature, label)
        
        all_feature = np.concatenate(all_feature, axis=0) #数据拼接
        all_label = np.concatenate(all_label, axis=0)
        data = {'feature': all_feature, 'label': all_label}
        np.save(f'{self.save_dir}/TotalAgent{self.num_agent}-AllData.npy', data)
    
    def save_data(self, agent_id, feature, label):
        data = {'feature': feature, 'label': label, 'theta_true': self.theta_true[agent_id], 'bias_true': self.bias[agent_id]}
        np.save(os.path.join(self.save_dir, f'TotalAgent{self.num_agent}-agent{agent_id}.npy'), data)

if __name__ == "__main__":
    num_agent = 30
    dim = 5
    num_local_data = 200
    save_dir = 'data/'

    data = CreateData(dim, num_agent, num_local_data, save_dir)
    data.generate_data()
    print("Data has been prepared !")