import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class Agent:
    """Interacts with and learns from the environment"""

    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, tau, lr, use_double_dqn, update_every,
                 device, seed):
        """
        Initialize an Agent.
        :param int state_size: Dimension of each state
        :param int action_size: Dimension of each action
        :param int buffer_size: Replay buffer size
        :param int batch_size: Mini batch size
        :param float gamma: Discount factor
        :param float tau: For soft update of target parameters
        :param float lr: Learning rate
        :param boolean use_double_dqn: Using double-learning strategy
        :param int update_every: How often to update the network
        :param torch.device device: GPU or CPU
        :param int seed: Random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.use_double_dqn = use_double_dqn
        self.update_every = update_every
        self.device = device
        random.seed(seed)

        # Q-Network
        self.q_network_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.q_network_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, eps=0.0):
        """
        Return actions for given state as per current policy.
        :param array_like state: current state
        :param float eps: epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """
        Agent makes a step in environment.
        :param int state:  Value of current state
        :param int action: Action chosen by the agent
        :param int reward: Reward after choosing action
        :param int next_state: Value of next state
        :param int done: Whether episode is finished
        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples
        Implementation of Double DQN comes from Miguel Morales' "Deep Reinforcement Learning".
        :param tuple[torch.Variable] experiences: tuple of (s, a, r, s', done) tuples
        :param float gamma: discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        if self.use_double_dqn:
            # Get argmax of predicted Q values (for next states) from target model
            argmax_a_q_sp = self.q_network_local(next_states).max(1)[1]

            q_sp = self.q_network_target(next_states).detach()
            max_a_q_sp = q_sp[np.arange(self.batch_size), argmax_a_q_sp].unsqueeze(1)
            max_a_q_sp *= (1 - dones)

            Q_targets = rewards + gamma * max_a_q_sp
        else:
            # Solution from course
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
            # Get expected Q values from local model; torch gather requires torch.LongTensor

        Q_expected = self.q_network_local(states).gather(1, actions.long())

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.q_network_local, self.q_network_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters - 0_target = tau * 0_local + (1 - tau) * Q_target
        Polyak averaging method to mix the target network with a tiny bit of the online network more frequently (?)
        Equation suggests that.
        :param PyTorch model local_model: weights will be copied from
        :param PyTorch model target_model: weights will be copied to
        :param float tau: interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
