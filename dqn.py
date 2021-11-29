from collections import deque

import numpy as np
import torch


def dqn(env, brain_name, result_dir, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning
    :param env: RL environment
    :param brain_name: Chosen brain
    :param Path result_dir: Path to result dir
    :param agent: Agent object
    :param int n_episodes: Maximum number of training episodes
    :param int max_t: Maximum number of timesteps per episode
    :param float eps_start: Starting value of epsilon, for epsilon-greedy action selection
    :param float eps_end: Minimum value of epsilon
    :param float eps_decay: Multiplicative factor (per episode) for decreasing epsilon
    :return list scores: Scores from each episode
    """

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)  # select action
            env_info = env.step(action)[brain_name]     # send the action to the environment
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        eps = max(eps_end, eps_decay*eps)   # decrease epsilon

        print(f'\rEpisode: {i_episode}\tAverage_score: {np.mean(scores_window)}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode: {i_episode}\tAverage_score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= 13.0:
            print(f'\nEnvironment solved in: {i_episode - 100} episodes!\tAverage_score: {np.mean(scores_window)}')
            torch.save(agent.q_network_local.state_dict(), result_dir / 'navigation_model_solution.pth')
            break
    return scores
