import numpy as np
from logistic import LogisticMin
from gd import GradientDescent

num_agent = 30
data_dir = './data'

problem = LogisticMin(num_agent, data_dir)
X, Y = problem.allfeature, problem.alllabel
theta = np.random.rand(problem.dim+1,) #problem.dim is feature num, since we add bias term, dim(theta) needs to be added by 1

gd = GradientDescent(problem, theta_init=theta)
gd.fit(X, Y, tol=1e-9, maxIter=1e5, show_flag=True)
theta_opt = np.copy(gd.theta)

print(f'theta opt is {theta_opt}')
np.save('./data/'+f"theta_opt.npy", theta_opt)