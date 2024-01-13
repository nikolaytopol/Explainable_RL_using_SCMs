# A Causality-based Method for Explanation Generation in Reinforcement Learning 

This framework generates causal explanations for StarCraft II agents using structural causal models.

StarCraft II agents are trained using reinforcement learning Advantage Actor-Critic (A2C) algorithm.
Custom maps can be added, with different state features and actions (current agent uses a custom version of the Simple64 map with 9 features and 4 actions).
Structural equations can be trained using 3 different regressors (linear, trees and neural).
Installation

-requires pysc2 3.0+ to run. Please follow the installation instructions at pysc2 repo to install StarCraft II game and maps. Code was tested in a Linux environment. In a Windows system, framework should still work, but pysc2 won't be able to run headless.

-requires Tensorflow 1.14 (only tested with this version).

-requires numpy and pandas.

-requires networkx 1.11 (only works with 1.11 version)

-after installing Starcraft II and the maps, copy (and replace) the custom Simple64 map given in this repo to the StarCraft II installation directory/StarCraftII/Maps/Melee/

Training and generating explanations

-To train with linear regressors for the Structural Causal Model (SCM)

$ python main.py test1 --map Simple64 --res 64 --envs 1 --max_windows 1 --nhwc --ow --data_size 2000 --scm_mode train --scm_regressor lr
Please see config_simulations.py for other configurations. Before training remove pre-trained SCM's in scm_model directory -To generate explanations