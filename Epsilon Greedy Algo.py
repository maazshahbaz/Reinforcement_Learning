import numpy as np

T = 100  # Total number of trials.
N = 4   # Number of machines.
r = 1   # Reward of a success trial.
epsilon = 0.1  # Define the parameter of epsilon for the epsilon-greedy algorithm.
p_a = np.array([0.3, 0.5, 0.2, 0.8])
Q_a = np.zeros(N)
N_a = np.zeros(N)
reward_matrix = np.zeros((N, T))
total_return = 0

for k in range(1, T + 1):
    p_exploration = np.random.rand()

    if p_exploration <= epsilon:  # Conduct exploration at this trial.
        a_k = np.random.randint(1, N + 1)  # Randomly selected action following uniform distribution.
        p_action_success = np.random.rand()
        N_a[a_k - 1] += 1

        if p_action_success < p_a[a_k - 1]:  # Trial successful.
            if k == 1:
                Q_a[a_k - 1] = (1 / N_a[a_k - 1]) * r
                reward_matrix[a_k - 1, k - 1] = r
                total_return += r
            else:
                reward_tmp = np.sum(reward_matrix[a_k - 1, :k - 1])
                Q_a[a_k - 1] = (1 / N_a[a_k - 1]) * (r + reward_tmp)
                reward_matrix[a_k - 1, k - 1] = r
                total_return += r
        else:  # Trial failed.
            if k == 1:
                Q_a[a_k - 1] = 0
            else:
                reward_tmp = np.sum(reward_matrix[a_k - 1, :k - 1])
                Q_a[a_k - 1] = (1 / N_a[a_k - 1]) * reward_tmp
    else:  # Select the action with the maximum action value at this trial.
        max_value = np.max(Q_a)
        max_indices = np.where(Q_a == max_value)[0]
        a_k = np.random.choice(max_indices) + 1
        p_action_success = np.random.rand()
        N_a[a_k - 1] += 1

        if p_action_success < p_a[a_k - 1]:  # Trial successful.
            if k == 1:
                Q_a[a_k - 1] = (1 / N_a[a_k - 1]) * r
                reward_matrix[a_k - 1, k - 1] = r
                total_return += r
            else:
                reward_tmp = np.sum(reward_matrix[a_k - 1, :k - 1])
                Q_a[a_k - 1] = (1 / N_a[a_k - 1]) * (r + reward_tmp)
                reward_matrix[a_k - 1, k - 1] = r
                total_return += r
        else:  # Trial failed.
            if k == 1:
                Q_a[a_k - 1] = 0
            else:
                reward_tmp = np.sum(reward_matrix[a_k - 1, :k - 1])
                Q_a[a_k - 1] = (1 / N_a[a_k - 1]) * reward_tmp

# Print the results or use them as needed.
print("Q_a:", Q_a)
print("Total Return:", total_return)
