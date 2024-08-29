import numpy as np 
from tqdm import tqdm
import os
from mpi4py import MPI

from gridworld import GridWorld
from agent import TD0Agent
from util import creat_graph, create_sampling_time


def run_simu():
    """
        用于测试单个agent的TD0算法
    """
    env = GridWorld()

    # 定义一个随机策略
    policy = np.ones((env.num_states, env.num_actions))
    policy /= np.sum(policy, axis=1, keepdims=True)  # 使每个状态的策略概率之和为1

    # 初始化一个 TDAgent
    agent_info = {
        'discount_factor': 0.9,
        'policy': policy,
        'num_states': env.num_states,
        'num_actions': env.num_actions
    }
    agent = TD0Agent(agent_info)

    num_episodes = int(1e5)

    for iter in tqdm(range(num_episodes)):
        if iter == 0:
            state = env.env_start()
            action = agent.agent_start(state)
        reward, state = env.step(action)
        action = agent.step(reward, state, iter)
    
    save_dir = './res'
    
    np.save(os.path.join(save_dir, 'res-agent0.npy'), agent.metric) 


class DecTD0(object):

    def __init__(self, num_agents, logMaxIter):
        num_states = 16
        num_actions = 4
        self.num_agents = num_agents
        self.policy = self.uniform_policy(num_states, num_actions)

        agent_info = {
            'discount_factor': 0.9,
            'policy': self.policy,
            'num_states': num_states,
            'num_actions': num_actions
        }
        self.agent_list = []
        self.agent_last_action = []
        self.env_list = []
        self.env_last_state = []

        for agent_id in range(num_agents):
            agent_info.update({'agent_id': agent_id})
            self.agent_list.append(TD0Agent(agent_info))
            self.env_list.append(GridWorld())
        
        # 其他参数
        self.num_episodes = int(10**logMaxIter)
        self.W = self.topo(num_agents)
        self.sample_num = create_sampling_time(logMaxIter, log_scale=True)
        self.metric = {'iter': [], 'dist2opt': [], 'worstdist2opt': []}

        # run in parallel
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
    
    def start(self):
        for agent_id in range(self.num_agents):
            state = self.env_list[agent_id].env_start()
            action = self.agent_list[agent_id].agent_start(state)
            # 第一步的action和state填充到列表中
            self.agent_last_action.append(action)
            self.env_last_state.append(state)
    
    def run(self):
        self.start()
        if self.rank == 0:
            loop = tqdm(range(self.num_episodes))
        else:
            loop = range(self.num_episodes)

        for iter in loop:
            self.consensus()
            for agent_id in range(self.num_agents):
                reward, state = self.env_list[agent_id].step(self.agent_last_action[agent_id])
                action = self.agent_list[agent_id].step(reward, state, iter)

                # 修改last_action和last_state
                self.agent_last_action[agent_id] = action
                self.env_last_state[agent_id] = state
            self.record(iter)
        self.save_info()
        print()
        print('Decentralized TD0 algorithm, Done!')
    
    def consensus(self):
        for i in range(self.num_agents):
            temp = np.zeros_like(self.agent_list[i].prm)
            for j in range(self.num_agents):
                temp += self.W[i, j] * self.agent_list[j].prm
            self.agent_list[i].prm = temp
    
    def topo(self, num_agents):
        self_weight = 0.8
        W = creat_graph(num_agents, 'RingGraph', self_weight)
        return W
    
    def record(self, iter):
        if iter in self.sample_num:
            self.metric['iter'].append(iter)
            avg_metric = []
            for agent_id in range(self.num_agents):
                avg_metric.append(self.agent_list[agent_id].metric['dist2opt'][-1])
            self.metric['worstdist2opt'].append(np.max(avg_metric))
            self.metric['dist2opt'].append(np.mean(avg_metric))
    
    def save_info(self):
        save_dir = './res'
        np.save(os.path.join(save_dir, f'res-rep{self.rank}-dectd.npy'), self.metric) 

    def uniform_policy(self, num_states, num_actions):
        policy = np.ones((num_states, num_actions))
        policy /= np.sum(policy, axis=1, keepdims=True)  # 使每个状态的策略概率之和为1
        return policy


class CenTD0(object):

    def __init__(self, num_agents, logMaxIter):
        num_states = 16
        num_actions = 4
        self.batch = num_agents
        self.policy = self.uniform_policy(num_states, num_actions)

        agent_info = {
            'discount_factor': 0.9,
            'policy': self.policy,
            'num_states': num_states,
            'num_actions': num_actions
        }
        
        self.env = GridWorld()
        self.env_last_state = None
        self.agent = TD0Agent(agent_info)
        self.agent_last_action = None
        
        # 其他参数
        self.num_episodes = int(10**logMaxIter)
        self.sample_num = create_sampling_time(logMaxIter, log_scale=True)
        self.metric = {'iter': [], 'dist2opt': []}

        # run in parallel
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
    
    def start(self):
            
        state = self.env.env_start()
        action = self.agent.agent_start(state)
        # 第一步的action和state填充到列表中
        self.agent_last_action = action
        self.env_last_state = state
    
    def run(self):
        self.start()

        if self.rank == 0:
            loop = tqdm(range(self.num_episodes))
        else:
            loop = range(self.num_episodes)
        for iter in loop:
            reward_list = []
            for _ in range(self.batch):
                reward, state = self.env.step(self.agent_last_action)
                reward_list.append(reward)

            action = self.agent.step(np.mean(reward_list), state, iter)
            # 修改last_action和last_state
            self.agent_last_action = action
            self.env_last_state = state
            self.record(iter)
        self.save_info()
        print()
        print('Centralized TD0 algorithm, Done!')
    
    def save_info(self):
        save_dir = './res'
        np.save(os.path.join(save_dir, f'res-rep{self.rank}-centd.npy'), self.metric) 

    def uniform_policy(self, num_states, num_actions):
        policy = np.ones((num_states, num_actions))
        policy /= np.sum(policy, axis=1, keepdims=True)  # 使每个状态的策略概率之和为1
        return policy

    def record(self, iter):
        if iter in self.sample_num:
            self.metric['iter'].append(iter)
            self.metric['dist2opt'].append(self.agent.metric['dist2opt'][-1])


if __name__ == '__main__':
    # run_simu()

    DecTD_algo = DecTD0(num_agents=10, logMaxIter=5)
    DecTD_algo.run()

    CenTD_algo = CenTD0(num_agents=10, logMaxIter=5)
    CenTD_algo.run()
