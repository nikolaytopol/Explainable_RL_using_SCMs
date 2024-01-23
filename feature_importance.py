



import torch
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.preprocessing import preprocess_obs


import argparse
from argparse import Namespace
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

from main import load_dataframe

dfn=load_dataframe("Training_datasets/Bipedal_walker_PPO_training_data.pkl")


import argparse
from argparse import Namespace
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


env_name = EnvironmentName("BipedalWalker-v3")
algo = "ppo"
folder = "rl-baselines3-zoo/rl-trained-agents/"

n_timesteps = 10

args = Namespace(
    env=env_name,
    folder=folder,
    algo=algo,
    n_timesteps=n_timesteps,
    num_threads=-1,
    n_envs=1,
    exp_id=0,
    verbose=1,
    no_render=False,
    deterministic=False,
    device="auto",
    load_best=False,
    load_checkpoint=None,
    load_last_checkpoint=False,
    stochastic=False,
    norm_reward=False,
    seed=0,
    reward_log="",
    gym_packages=[],
    env_kwargs=None,
    custom_objects=False,
    progress=False
)

# Going through custom gym packages to let them register in the global registory
for env_module in args.gym_packages:
    importlib.import_module(env_module)

env_name: EnvironmentName = args.env
algo = args.algo
folder = args.folder

try:
    _, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )
except (AssertionError, ValueError) as e:
    # Special case for rl-trained agents
    # auto-download from the hub
    if "rl-trained-agents" not in folder:
        raise e
    else:
        print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
        # Auto-download
        download_from_hub(
            algo=algo,
            env_name=env_name,
            exp_id=args.exp_id,
            folder=folder,
            organization="sb3",
            repo_name=None,
            force=False,
        )
        # Try again
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )

print(f"Loadings {folder}")

# Off-policy algorithm only support one env for now
off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

if algo in off_policy_algos:
    args.n_envs = 1

set_random_seed(args.seed)

if args.num_threads > 0:
    if args.verbose > 1:
        print(f"Setting torch.num_threads to {args.num_threads}")
    th.set_num_threads(args.num_threads)

is_atari = ExperimentManager.is_atari(env_name.gym_id)

stats_path = os.path.join(log_path, env_name)
hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

# load env_kwargs if existing
env_kwargs = {}
args_path = os.path.join(log_path, env_name, "args.yml")
if os.path.isfile(args_path):
    with open(args_path) as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]
# overwrite with command line arguments
if args.env_kwargs is not None:
    env_kwargs.update(args.env_kwargs)

log_dir = args.reward_log if args.reward_log != "" else None

env = create_test_env(
    env_name.gym_id,
    n_envs=args.n_envs,
    stats_path=maybe_stats_path,
    seed=args.seed,
    log_dir=log_dir,
    should_render=not args.no_render,
    hyperparams=hyperparams,
    env_kwargs=env_kwargs,
)


kwargs = __builtins__.dict(seed=args.seed)
if algo in off_policy_algos:
    # Dummy buffer size as we don't need memory to enjoy the trained agent
    kwargs.update(dict(buffer_size=1))
    # Hack due to breaking change in v1.6
    # handle_timeout_termination cannot be at the same time
    # with optimize_memory_usage
    if "optimize_memory_usage" in hyperparams:
        kwargs.update(optimize_memory_usage=False)

# Check if we are running python 3.8+
# we need to patch saved model under python 3.6/3.7 to load them
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

custom_objects = {}
if newer_python_version or args.custom_objects:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, device=args.device, **kwargs)
#model.save('/Users/userok/Desktop/research_project/Report_writing/Bipedal_walker_PPO/PPO_model.zip')

obs = env.reset()

# Deterministic by default except for atari games
stochastic = args.stochastic or is_atari and not args.deterministic
deterministic = not stochastic

episode_reward = 0.0
episode_rewards, episode_lengths = [], []
ep_len = 0
# For HER, monitor success rate
successes = []
lstm_states = None
episode_start = np.ones((env.num_envs,), dtype=bool)

generator = range(args.n_timesteps)
if args.progress:
    if tqdm is None:
        raise ImportError("Please install tqdm and rich to use the progress bar")
    generator = tqdm(generator)

try:
    import pandas as pd
    # Initialize DataFrame/////
    log_df = pd.DataFrame(columns=["state", "action", "reward", "done"])
    

    for _ in generator:
        print("Observation dtype:", obs.dtype)
        if lstm_states is not None:
            print("LSTM states dtype:", lstm_states.dtype)
        action, lstm_states = model.predict(
            obs,  # type: ignore[arg-type]
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
    
        obs, reward, done, infos = env.step(action)

        # Log data
        new_row = {'state': obs[0], 'action': action[0], 'reward': reward[0], 'done': done[0]}
        log_df = log_df._append(new_row, ignore_index=True)

        episode_start = done

        if not args.no_render:
            pass #env.render("human")

        episode_reward += reward[0]
        ep_len += 1

        if args.n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get("episode")
                if episode_infos is not None:
                    print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                    print("Atari Episode Length", episode_infos["l"])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print(f"Episode Reward: {episode_reward:.2f}")
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_reward = 0.0
                ep_len = 0

            # Reset also when the goal is achieved when using HER
            if done and infos[0].get("is_success") is not None:
                if args.verbose > 1:
                    print("Success?", infos[0].get("is_success", False))

                if infos[0].get("is_success") is not None:
                    successes.append(infos[0].get("is_success", False))
                    episode_reward, ep_len = 0.0, 0

    # After the loop ends /////

    #log_df.to_pickle("/Users/userok/Desktop/research_project/Report_writing/Bipedal_walker_PPO/Bipedal_walker_PPO_training_data.pkl")
    # Save DataFrame to a CSV file


except KeyboardInterrupt:
    # After the loop ends /////
    #log_df.to_pickle("/Users/userok/Desktop/research_project/Report_writing/Bipedal_walker_PPO/Bipedal_walker_PPO_training_data.pkl")
    # Save DataFrame to a CSV file
    pass

if args.verbose > 0 and len(successes) > 0:
    print(f"Success rate: {100 * np.mean(successes):.2f}%")

if args.verbose > 0 and len(episode_rewards) > 0:
    print(f"{len(episode_rewards)} Episodes")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

if args.verbose > 0 and len(episode_lengths) > 0:
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

env.close() 




import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.preprocessing import preprocess_obs

# Define the function to calculate SHAP values
def calculate_shapley_values(model, df):
    # Preprocess the observations
    observations = dfn.iloc[:, :24].values.astype(np.float32)
    observations = torch.tensor(observations)
    observations = preprocess_obs(observations, model.observation_space)

    # Get the model's predicted action values for the observations
    action_values = model.predict(observations)[0]


    # Calculate the SHAP values for each feature
    num_features = observations.shape[1]
    shapley_values = np.zeros(num_features)

    for i in range(num_features):
        # Create a copy of the observations with the feature set to 0
        observations_copy = observations.clone()
        observations_copy[:, i] = 0

        # Get the model's predicted action values for the modified observations
        action_values_without_feature = model.predict(observations_copy)[0]

        # Calculate the SHAP value for the current feature
        shapley_value = np.mean(action_values - action_values_without_feature)
        
        

        # Add the SHAP value to the array
        shapley_values[i] = shapley_value

    return shapley_values


# Calculate the SHAP values
shapley_values= calculate_shapley_values(model, dfn)

# Sort the features and SHAP values in descending order
feature_names = dfn.iloc[:, :24].columns
sorted_indices = np.argsort(-shapley_values)
sorted_features = feature_names[sorted_indices]
sorted_shapley_values = shapley_values[sorted_indices]

# Create the summary plot
plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_features)), sorted_shapley_values)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('SHAP Value')
plt.ylabel('Feature')
plt.title('Summary Plot: SHAP Values Impact on Model Output')
# Save the plot as a high-resolution picture
plt.savefig("Bipedal_walker_SHAP_values.jpg", dpi=300)
plt.show()

# Selecting 9 most important state sapce vairables
new_indices=np.argsort(-abs(shapley_values))
new_sorted_features=feature_names[new_indices]
first_nine=new_sorted_features[:10]
print('Most influential variables are :',first_nine)
