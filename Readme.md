## Overview

This repository contains the code for numerical experiments related to my paper ``Tighter Analysis for Decentralized Stochastic Gradient Method: Impact of Data Homogeneity" [Arxiv link](https://arxiv.org/abs/2409.04092). 

The experiments focus on decentralized stochastic gradient descent (DSGD) and illustrate the theoretical findings on the effect of data homogeneity on the convergence of the algorithm.

## Structure

The repository is organized as follows:

- **Quadratic Stochastic Optimization** (`quadratic/`): 
  - Code implementing the quadratic stochastic optimization problem where agents minimize a global objective function composed of local quadratic terms.
  - Includes scripts for generating synthetic data, solving the optimization problem, and evaluating performance metrics.

- **Logistic Regression** (`logistic/`): 
  - Contains the implementation of a binary classification problem using logistic regression with decentralized training across 25 agents.
  - Includes data generation, training, and evaluation scripts for both ring and Erdos-Renyi (ER) graph topologies.

- **Decentralized TD(0) Learning** (`multi-agent-TD/`): 
  - Implements a decentralized TD(0) learning algorithm for policy evaluation in a GridWorld environment.
  - Features scripts for setting up the GridWorld environment, running the decentralized TD(0) algorithm, and analyzing the convergence behavior.

## Requirements

The code requires the following Python packages:
- `numpy`, `scipy`, `matploblib`, `networkx`, `mpi4py`

## Citation

Please consider citing out paper if you use our code:

<div style="position: relative;">
    <pre>
@misc{li2024tighter,
      title={Tighter Analysis for Decentralized Stochastic Gradient Method: Impact of Data Homogeneity}, 
      author={Qiang Li and Hoi-To Wai},
      year={2024},
      eprint={2409.04092},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2409.04092}, 
}
    </pre>
    <button style="position: absolute; top: 10px; right: 10px;" onclick="copyToClipboard()">Copy</button>
</div>

If you find this repository helpful, don't hesitate to give me a star ⭐! Sending love 💖💖!
