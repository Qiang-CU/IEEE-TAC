import numpy as np
from tqdm import tqdm


# This file is used to get the optimal 

def get_mu(policy, env, tol=1e-4):
    
    num_states, num_actions = policy.shape
    P = np.zeros((num_states, num_actions, num_states))
    trans_mat = np.zeros((num_states, num_states))
    num_trails = int(1e4)

    for _ in tqdm(range(num_trails)):
        for state in range(num_states):
            action = np.random.choice(num_actions, p=policy[state])
            next_state = env.move(state, action)
            P[state, action, next_state] += 1

    for state in range(num_states):
        for next_state in range(num_states):
            trans_mat[state, next_state] = P[state, :, next_state].sum() / num_trails
    
    # 初始值，可以选择均匀分布或者任意非零分布
    initial_distribution = np.ones(num_states) / num_states

    # 设置迭代次数和收敛阈值
    max_iterations = int(5 * 1e4)

    for _ in range(max_iterations):
        old_distribution = initial_distribution.copy()
        initial_distribution = np.dot(initial_distribution, trans_mat)
        initial_distribution /= initial_distribution.sum()  # 归一化新分布
        if np.abs(initial_distribution - old_distribution).sum() < tol:
            break

    # 输出平稳分布
    # print("Stationary distribution: ", initial_distribution)
    return initial_distribution, trans_mat

def get_optprm(mu, trans_mat, reward, env, agent):

    lhs = np.zeros((agent.dim, agent.dim))
    for s in range(env.num_states):
        tmp = 0
        for next_s in range(env.num_states):
            tmp += agent.feature[next_s] * trans_mat[s][next_s]
        lhs += mu[s] * np.outer(agent.feature[s], (agent.feature[s] - agent.discount_factor * tmp))
        
    rhs = 0
    for s in range(env.num_states):
        rhs += mu[s] * agent.feature[s] * reward[s]
    
    optprm = np.linalg.solve(lhs, rhs)

    return optprm
    

if __name__ == "__main__":
    num_actions = 4
    num_states = 16
    policy = np.ones((num_states, num_actions)) / num_actions 

    from gridworld import GridWorld
    from agent import TD0Agent
    env = GridWorld()
    agent_info = {
        'discount_factor': 0.9,
        'policy': policy,
        'num_states': num_states,
        'num_actions': num_actions
    }
    agent = TD0Agent(agent_info)
    reward = [np.random.normal(1,0) for i in range(env.num_states)]

    mu, trans_mat = get_mu(policy, env)
    optprm = get_optprm(mu, trans_mat, reward, env, agent)

    np.save('optprm.npy', optprm)

    print(optprm)
