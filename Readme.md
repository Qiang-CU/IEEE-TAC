## Overview

This repository contains the code for numerical experiments related to my subbmitted paper. The experiments focus on decentralized stochastic gradient descent (DSGD) and illustrate the theoretical findings on the effect of data homogeneity on the convergence of the algorithm.

## Structure

The repository is organized as follows:

- **Quadratic Stochastic Optimization** (`quadratic_optimization/`): 
  - Code implementing the quadratic stochastic optimization problem where agents minimize a global objective function composed of local quadratic terms.
  - Includes scripts for generating synthetic data, solving the optimization problem, and evaluating performance metrics.

- **Logistic Regression** (`logistic_regression/`): 
  - Contains the implementation of a binary classification problem using logistic regression with decentralized training across 25 agents.
  - Includes data generation, training, and evaluation scripts for both ring and Erdos-Renyi (ER) graph topologies.

- **Decentralized TD(0) Learning** (`td_learning/`): 
  - Implements a decentralized TD(0) learning algorithm for policy evaluation in a GridWorld environment.
  - Features scripts for setting up the GridWorld environment, running the decentralized TD(0) algorithm, and analyzing the convergence behavior.

## Requirements

The code requires the following Python packages:
- `numpy`, `scipy`, `matploblib`, `networkx`, `mpi4py`