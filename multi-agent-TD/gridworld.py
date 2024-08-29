import numpy as np 


"""
grid world env described by no communication paper in page 27
"""

# 顺时针方向定义action
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld(object):

    def __init__(self):
        self.num_col = 4
        self.num_row = 4
        self.num_states = self.num_col * self.num_row
        self.num_actions = 4
        self.states = np.arange(self.num_states)

        self.cur_state = 0 #初始状态设置为0

    
    def state2cord(self, state):
        return [state // self.num_col, state % self.num_col]
    
    def cord2state(self, cord):
        return cord[0] * self.num_col + cord[1]
    
    def move(self, state, action):
        cord = self.state2cord(state) 
        next_cord = cord.copy()
        if action == UP:
            next_cord[0] = max(0, cord[0] - 1)
        elif action == RIGHT:
            next_cord[1] = min(self.num_col - 1, cord[1] + 1)
        elif action == DOWN:
            next_cord[0] = min(self.num_row - 1, cord[0] + 1)
        elif action == LEFT:
            next_cord[1] = max(0, cord[1] - 1)
        else:
            raise ValueError("Action not defined")
        
        next_state = self.cord2state(next_cord)
        return next_state
    
    def env_start(self):
        state = self.cur_state
        observation = [0, state]
        return state

    def step(self, action):
        """
        action: 0, 1, 2, 3
        """
        
        next_state = self.move(self.cur_state, action)
        self.cur_state = next_state
        
        reward = np.random.normal(1,20) # 定义环境给出的reward r(s,a)是一个服从正态分布的随机变量
        
        observation = [reward, next_state]
        return observation

if __name__ == '__main__':
    env = GridWorld()
    env.cur_state = 12
    print(env.step(LEFT))
    print(env.step(RIGHT))
    print(env.step(DOWN))
    print(env.step(UP))