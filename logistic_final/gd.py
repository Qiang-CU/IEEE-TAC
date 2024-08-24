import numpy as np 
import time

class GradientDescent(object):
    def __init__(self, problem, save_flag = False, theta_init=None):
        self.problem = problem
        if theta_init is None:
            self.theta = np.zeros(self.problem.d)
        else: 
            self.theta = theta_init
        self.save_flag = save_flag
        self.measurement = {"iter": [], "theta": [], 'loss': []}
    
    def record(self, iter, theta, loss):
        self.measurement['iter'].append(iter)
        self.measurement['theta'].append(theta)
        self.measurement['loss'].append(loss)
    
    def save(self):
        path = './res_gd/'
        np.save(path+f"gd.npy", self.measurements)
    
    def fit(self, X, Y, tol=1e-7, maxIter=1e3, show_flag=False):
        n = X.shape[0]
        gap = 1e30
        count = 0
        smoothness = np.sum(np.square(np.linalg.norm(X, axis=1))) / (4.0 * n) + self.problem.lam
        eta_init = 1.0 / smoothness
        eta = eta_init
        pre_loss = self.problem.loss(X, Y, self.theta)
        self.record(0, self.theta, pre_loss)

        while count < maxIter:
            if gap < tol:
                break

            grd = self.problem.grad(X, Y, self.theta)
            theta_new = self.theta - eta * grd
            
            if eta < 1e-10:
                break

            new_loss = self.problem.loss(X, Y, theta_new)

            if new_loss > pre_loss:
                eta = eta * 0.1
                gap = 1e30
                continue
            else:
                eta = eta_init
            
            self.theta = np.copy(theta_new)
            gap = pre_loss - new_loss
            pre_loss = new_loss
            count = count + 1
            self.record(count, self.theta, new_loss)

            if show_flag and count % 10 == 0:
                print(f'Grd Descent: iter: {count}, loss: {new_loss:.4e}, gap: {gap :.5e}')
        

