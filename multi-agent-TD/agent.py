import numpy as np

"""
实现：可以执行TD(0)算法的一个agent, with linear function approximation
"""


class TD0Agent(object):
    def __init__(self, agent_info):
        self.discount_factor = agent_info.get('discount_factor')
        self.policy = agent_info.get('policy') # should be a matrix: num_states * num_actions
        self.num_states = agent_info.get('num_states')
        self.num_actions = agent_info.get('num_actions')
        self.agent_id = agent_info.get('agent_id')
        self.dim = 4 # number of features for each state

        self.num_actions = 4
        self.pre_state = None
        self.pre_action = None

        self.feature = self.create_feature()
        self.prm = np.zeros((self.dim)) # linear function approximation parameter
        self.metric = {'iter': [], 'dist2opt': []}
    
    def create_feature(self):
        # 只针对16个状态的gridworld设计的特征
        feature = np.zeros((self.num_states, self.dim))
        for i in range(self.num_states):
            if i in [0,1,4,5]:
                feature[i] = np.array([1,0,0,0])
            elif i in [2,3,6,7]:
                feature[i] = np.array([0,1,0,0])
            elif i in [8,9,12,13]:
                feature[i] = np.array([0,0,1,0])
            else:
                feature[i] = np.array([0,0,0,1])
        return feature
    
    def step_size(self, iter):
        a0 = 100
        a1 = 200
        return a0 / (a1 + iter)
    
    def agent_start(self, state):
        """
        agent第一次与环境交互时调用，state是环境给出的初始状态
        """
        action = np.random.choice(self.num_actions, p=self.policy[state])
        self.prev_state = state
        self.prev_action = action
        return action

    def step(self, reward, state, iter):
        """
        reward: 从环境中得到的上一步动作的reward
        state: 从环境中得到的上一步动作后的状态
        iter: 当前迭代次数
        """
        action =  np.random.choice(self.num_actions, p=self.policy[state]) 

        # perform once TD(0) update
        grd = self.feature[self.prev_state] * (reward + np.dot(self.discount_factor * self.feature[state] - self.feature[self.prev_state], self.prm))

        self.prm += self.step_size(iter) * grd

        self.prev_state = state
        self.prev_action = action

        self.record(iter)
        return action
    
    def record(self, iter):
        if iter == 0:
            self.optprm = np.load('optprm.npy')
        self.metric['iter'].append(iter)
        self.metric['dist2opt'].append(np.linalg.norm(self.optprm - self.prm, ord=2)**2)
        
    
if __name__ == "__main__":
    
    # 定义一个简单的环境，假设状态空间为 [0, 1, 2]，动作空间为 [0, 1, 2, 3]
    num_states = 16
    num_actions = 4

    # 定义一个随机策略
    policy = np.random.rand(num_states, num_actions)
    policy /= np.sum(policy, axis=1, keepdims=True)  # 使每个状态的策略概率之和为1

    # 初始化一个 TDAgent
    agent_info = {
        'discount_factor': 0.9,
        'policy': policy,
        'num_states': num_states,
        'num_actions': num_actions
    }
    agent = TD0Agent(agent_info)

    # 在初始状态下获取一个动作
    state = 0
    action = agent.agent_start(state)
    print("Selected action in state", state, ":", action)

    # 进行若干步迭代
    num_iterations = 10
    for i in range(num_iterations):
        # 模拟环境给出的奖励和下一个状态
        reward = np.random.rand()  # 模拟一个随机奖励
        next_state = np.random.choice(num_states)  # 模拟一个随机的下一个状态

        # 让代理执行一步
        action = agent.step(reward, next_state, i)

        print("Iteration:", i+1)
        print("Reward:", reward)
        print("Next state:", next_state)
        print("Selected action in next state:", action)
        print("Parameter prm:", agent.prm)
        print()
