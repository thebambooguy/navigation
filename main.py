import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from unityagents import UnityEnvironment

from dqn import dqn
from dqn_agent import Agent

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_and_parse_args(args=None):
    '''
    :param args: A list of arguments as if they were input in the command line. Leave it None to use sys.argv
    '''

    parser = argparse.ArgumentParser(description='Navigation - train agent to collect yellow bananas')
    parser.add_argument("--config_file", type=Path, default="config.yaml", help='Path to config file')
    parser.add_argument("--results_dir", type=Path, default="results", help='Path to results dir')
    args = parser.parse_args(args)
    return args


def get_config(path):
    print(f'Reading config file from {path}')
    with path.open() as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == '__main__':
    args = create_and_parse_args()
    config = get_config(args.config_file)

    env = UnityEnvironment(file_name=config['ENV_FILE'])

    # Environments contain brains which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print(f'Number of agents: {len(env_info.agents)}')
    # number of actions
    action_size = brain.vector_action_space_size
    print(f'Number of actions: {action_size}')
    # examine the state space
    state = env_info.vector_observations[0]
    print(f'States look like: {state}')
    state_size = len(state)
    print(f'States have length: {state_size}')

    agent = Agent(state_size=state_size, action_size=action_size, buffer_size=config['BUFFER_SIZE'],
                  batch_size=config['BATCH_SIZE'], gamma=config['GAMMA'], tau=config['TAU'], lr=config['LR'],
                  use_double_dqn=config['USE_DOUBLE_DQN'], update_every=config['UPDATE_EVERY'], device=DEVICE,
                  seed=config['SEED'])

    date = datetime.today().strftime('%Y-%m-%d-%H-%M')
    save_dir = args.results_dir / date
    save_dir.mkdir(exist_ok=True, parents=True)

    scores = dqn(env, brain_name, save_dir, agent, n_episodes=config['N_EPISODES'], max_t=config['MAX_T'],
                 eps_start=config['EPS_START'], eps_end=config['EPS_END'], eps_decay=config['EPS_DECAY'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(save_dir / 'agent_scores.png')

    with open(save_dir / f'{date}_config.yaml', 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    env.close()
