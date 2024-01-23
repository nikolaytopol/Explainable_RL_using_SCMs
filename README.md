# A Causality-Based Method for Explanation Generation in Reinforcement Learning

This repository contains the implementation of the CaM-XRL framework, designed to generate causal explanations for reinforcement learning agents.

## Feature Importance Methodology for Bipedal Walker
**File: feature_importance.py**
- Identifies the most important variables in the dataset.
- Creates a file named "Bipedal_walker_SHAP_values.jpg" showing SHAP values.
- Prints the names of the most influential variables for further analysis.

## Algorithm Selection
**File: comparing_algorithms.py**
The algorithm selection process is implemented in the code. To initiate this process in the main code, please uncomment lines 146-149. Then, inspect the acquired Directed Acyclic Graphs (DAGs) and choose the algorithm that ensures the most stable causal discovery. Make sure influential variables are included in the DAG. Select this algorithm in the `main.py` file at lines ....

## Clustering
**File: clustering.py**
- Implements clustering method.
- Initially finds the optimal number of clusters using the elbow plot method.
- After inspecting the elbow plot, choose the optimal number of clusters by modifying the number at line 58 of `clustering.py`.
- Change the path in `clustering.py` to the environment you wish to explain.

## Instructions

### Data Selection
- Change the `file_name` variable to choose a specific dataset.
  - `dfn=load_dataframe(path)` - line 121
  - `file_name = "Cartpole_A2C_training_data.pkl"` - line 117

### Configurations
- Change the regressor type at line 399: 
  - `'function': get_regressor(x_feature_cols, key,'mlp')` # Select regressor here
- Choose the causal discovery algorithm type at line 193: 
  - `algorithm = GOLEM()` # Example: GOLEM

### Measuring faithfulness 
- Code at lines 486 - 498, measure faitfulness of the resultant model cahnge it to get higher test sample

### Note
- After re-launching the same environment, ensure to delete the previous SCM_models ile.

## Requirements
- Clone the repository with trained agents: git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo

- Install dependencies from the `packages.txt` file.

## Deployment

### Generating Datasets for Various Agents
- **Bipedal Walker PPO Agent**
- Command: 
  ```
  python ./helpers/agent_deployment/enjoy_BipedWalker.py --algo ppo --env BipedalWalker-v3 --folder rl-baselines3-zoo/rl-trained-agents/ -n 16
  ```
- Status: Done (Needs to be launched in the root directory of the repository)
- **Mountain Car DQN Agent**
- Command:
  ```
  python ./helpers/agent_deployment/enjoy_MontainCar.py --algo dqn --env MountainCar-v0 --folder rl-baselines3-zoo/rl-trained-agents/ -n 6000
  ```
- **Cartpole A2C Agent**
- Command:
  ```
  python ./helpers/agent_deployment/enjoy_Cartpole.py --algo a2c --env CartPole-v1 --folder rl-baselines3-zoo/rl-trained-agents/ -n 5000
  ```
- **Pendulum ARC Agent**
- Command:
  ```
  python ./helpers/agent_deployment/enjoy_Pendulum.py --algo sac --env Pendulum-v1 --folder rl-baselines3-zoo/rl-trained-agents/ -n 10000
  ```

- **Lunar Lander DQN Agent**
- Command:
  ```
  python ./helpers/agent_deployment/enjoy_LunarLander.py --algo dqn --env LunarLander-v2 --folder rl-baselines3-zoo/rl-trained-agents/ -n 10000
  ```



