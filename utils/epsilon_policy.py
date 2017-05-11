import numpy as np

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    Returns:
        A function that takes the state as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(state):
        # 1 / epsilon for non-greedy actions
        probs = (epsilon / nA) * np.ones(nA)

        greedy_action = Q[state].argmax()
        # (1 / epsilon + (1 - epsilon)) for greedy action
        probs[greedy_action] += 1.0 - epsilon

        return probs

    return policy_fn
