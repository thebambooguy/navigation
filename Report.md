[//]: # (Image References)


[fixed_q_targets]: ./images/fixed_q_targets.PNG
[best_agent_score]: ./images/best_agent_score.png

# Navigation project's report

## Learning algorithm

As a learning algorithm, a Deep Q-network was chosen. It is a next step into Reinforcement Learning
algorithms. Deep Q-network does not use Q-table anymore, as in traditional Q-learning and in contrast to Q-Learning,
it uses neural network to train the agent and which predicts action values for each possible environment action. 
However, using RL with neural network can be unstable. To deal with this instability, two functionalities are often
being implemented.

- **Experience Replay** - we do not want to learn from each of experience tuples `(S, A, R, S')` in sequential order
because they can be correlated, and we would like to avoid that. So, we store all experiences in a replay buffer and
uniformly sample a small batch from it in order to learn.
- **Fixed Q-Targets** - when calculating the loss, we calculate the difference between the TD target (Q_target) and 
the current Q value (estimation of Q), what means that we update a guess with a guess. We do not know what is the real
value of the TD target. In order to not chase a moving target, we use a separate target network with a fixed parameter
for estimating the TD target. The target Q-Network's weight are updated less often (or more slowly) than the 
primary Q-network.

    ![Fixed Q-targets equation][fixed_q_targets]

Implementation of Fixed Q-targets and solution with two separate networks, allows to introduce one from many available 
improvements to DQN called **Double Deep Q-network**. Our algorithm is an example of Deep Q-Learning, so similarly as in
Q-learning, we will always choose maximal estimated value as a next action. The problem is that we prefer higher values
more even if they aren't correct (especially at the beginning). With double DQN, we use one network to choose the best
action and the second one to evaluate that action. They need to sort og agree upon the best action.


Architecture of neural network:

In the project following architecture was built:
- Input layer: 37 neurons (size of the state space)
- Hidden layer #1: 64 neurons
- Output layer: 4 neurons (size of the action space)

Changing the number of neurons in hidden layer does not improve performance as well as adding another hidden layer.

When it comes to hyper-parameters, different scenarios were checked:
- UPDATE_EVERY (checked: 4, 8, 16) - best: 4
- GAMMA (checked: 0,95, 0.97, 0.95) - best: 0.95
- LR (checked: 0.0001, 0.0005, 0.001, 0.005) - best: 0.0005
- TAU (checked: 0.001, 0.005, 0.01) - best: 0.001


## Plot of Rewards

In order to solve the project, agent's performance must achieve average reward (over 100 episodes) of at least +13.
In this project, two solutions were tested: vanilla Deep Q-network and double deep Q-network. The second solution 
has a better performance and solve the environment ~70 episodes earlier (faster). Plot of rewards can be found below:


![Plot of rewards][best_agent_score]


The best agent was able to solve the environment in 368 episodes and this model was saved and uploaded to repo.
The worst agent solved the environment in 663 episodes.

## Ideas for Future Work

To improve the agent's performance, following ideas can be implemented:
- Using a different loss function - **Huber loss** 
- Choosing right **initialization strategy**
- Implementing a **dueling DQN**
- Taking advance of **Prioritized experience replay**