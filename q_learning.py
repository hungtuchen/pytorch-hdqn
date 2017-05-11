import numpy as np
import random
from collections import defaultdict
import itertools

from utils import plotting
from utils.epsilon_policy import make_epsilon_greedy_policy
from utils.schedule import LinearSchedule

def q_learning(env, num_episodes, discount_factor=1.0, lr=0.00025, exploration_schedule=LinearSchedule(50000, 0.1, 1.0)):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    Args:
        env: OpenAI environment.
        num_episodes: Number (can be divided by 1000) of episodes to run for. Ex: 12000
        discount_factor: Lambda time discount factor.
        lr: TD learning rate.
        exploration_schedule: Schedule (defined in utils.schedule)
            schedule for probability of chosing random action.
    Returns:
        A tuple (Q, stats, visits).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        visits is an 2D-array indicating how many time each state being visited in every 1000 episodes.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    # Keep track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    n_thousand_episode = int(np.floor(num_episodes / 1000))

    visits = np.zeros((n_thousand_episode, env.nS))

    total_timestep = 0

    for i_thousand_episode in range(n_thousand_episode):
        for i_episode in range(1000):
            current_state = env.reset()

            visits[i_thousand_episode][current_state-1] += 1
            # Keep track number of time-step per episode only for plotting
            for t in itertools.count():
                total_timestep += 1
                # Get annealing exploration rate (epislon) from exploration_schedule
                epsilon = exploration_schedule.value(total_timestep)
                # Improve epsilon greedy policy using lastest updated Q
                policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)

                # Choose the action based on epsilon greedy policy
                action_probs = policy(current_state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)

                visits[i_thousand_episode][next_state-1] += 1

                # Use the greedy action to evaluate Q, not the one we actually follow
                greedy_next_action = Q[next_state].argmax()
                # Evaluate Q using estimated action value of (next_state, greedy_next_action)
                td_target = reward + discount_factor * Q[next_state][greedy_next_action]
                td_error = td_target - Q[current_state][action]
                Q[current_state][action] += lr * td_error

                # Update statistics
                stats.episode_rewards[i_thousand_episode*1000 + i_episode] += reward
                stats.episode_lengths[i_thousand_episode*1000 + i_episode] = t

                if done:
                    break
                else:
                    current_state = next_state

    return Q, stats, visits
