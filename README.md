
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![Pytorch 1.3](https://img.shields.io/badge/pytorch-1.3.1-blue.svg)](https://pytorch.org/)

Theoretical findings have shown that, in suitably defined large data limit, **BNNs posteriors are robust to gradient-based adversarial
attacks**.Thus, this study aims to demonstrate the theoretical robustness of Bayesian
neural architectures against multiple white-box attacks and list empirical findings
from the same.

Part of this work has been inspired by @kumar-shridhar github:PyTorch-BayesianCNN which was used to estimate robustness against five state-of-the-art Gradient-based attacks: - 
```math
l_{\infty}-FGSM,\ l_{\infty}-PGD,\ l_{2}-PGD,\ BIM
```


### Layer types

This repository contains two types of bayesian lauer implementation:  
* **BBB (Bayes by Backprop):**  
  Based on [this paper](https://arxiv.org/abs/1505.05424). This layer samples all the weights individually and then combines them with the inputs to compute a sample from the activations.

* **BBB_LRT (Bayes by Backprop w/ Local Reparametrization Trick):**  
  This layer combines Bayes by Backprop with local reparametrization trick from [this paper](https://arxiv.org/abs/1506.02557). This trick makes it possible to directly sample from the distribution over activations.

#### Bayesian

`python main_bayesian.py`
* set hyperparameters in `config_bayesian.py`


#### Frequentist

`python main_frequentist.py`
* set hyperparameters in `config_frequentist.py`


### Directory Structure:
`layers/`:  Contains `ModuleWrapper`, `FlattenLayer`, `BBBLinear` and `BBBConv2d`.  
`models/BayesianModels/`: Contains standard Bayesian models (BBBLeNet, BBBAlexNet, BBB3Conv3FC).  
`models/NonBayesianModels/`: Contains standard Non-Bayesian models (LeNet, AlexNet).  
`checkpoints/`: Checkpoint directory: Models will be saved here.  
`tests/`: Basic unittest cases for layers and models.  
`main_bayesian.py`: Train and Evaluate Bayesian models.  
`config_bayesian.py`: Hyperparameters for `main_bayesian` file.  
`main_frequentist.py`: Train and Evaluate non-Bayesian (Frequentist) models.  
`config_frequentist.py`: Hyperparameters for `main_frequentist` file.  
`AttackingModels.ipynb` : Attacks performed on BNNs and plot generated

A detailed Report [here](Robustness_Report.pdf) written as a part of course project for CS690- Deep Learning Course at Purdue University is given.


