[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation

## Introduction

Navigation is a first project of a Deep Reinforcement Learning Nanodegree Program. For this project, an agent must be 
trained to navigate (and collect bananas!) in a large, square world. However, not all bananas should be collected. 
Agent should be careful and collect only yellow bananas!

![Trained Agent][image1]

## Project Details

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around
the agent's forward direction. Given this information, agent has to learn how to best select actions. Four discrete 
action are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

A reward of +1 is provided for collecting a yellow banana and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The task is episodic and in order to solve the environment, your agent must get an average score of +13 over 100 
consecutive episodes.

## Getting started
- **Step 1** - Set up your Python environment
  - Python 3.8 was used in the project
  - type `pip install -r requirements.txt` in command line to install requirements
- **Step 2** - Unzip `Banana_Linux.zip` environment
  - Note that provided environment will work only on Linux based machines!
- **Step 3** - Check `main.py` and `config.yaml` to get familiarized with arguments and hyper-parameters.
  
## Instruction
  In order to train agent, one should run `main.py` with proper arguments. The most important one is a `config_file`
  where path to Unity Environment, number of episodes or other hyper-parameters are defined. 
  Before running script, one should check exemplary `config.yaml` or create a new one. Plot with the agent's scores,
  used config and torch model (only if the environment was solved) will be stored in predefined result directory.
