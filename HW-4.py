import numpy as np

n = 7  # number of states

P = np.array([[0, 0.3, 0, 0, 0.2, 0.5, 0],
              [0, 0, 0.8, 0, 0, 0.2, 0],
              [0, 0, 0, 0.6, 0.3, 0, 0.1],
              [0, 0, 0, 0, 0, 0, 1],
              [0.2, 0.3, 0.5, 0, 0, 0, 0],
              [0.1, 0.1, 0, 0, 0, 0.8, 0],
              [0, 0, 0, 0, 0, 0, 1]])  # State transition matrix

I = np.eye(n)  # Identity matrix
gamma = 0.1  # gamma

r = np.array([-1, -2, -3, 10, -1, -4, 0])  # reward vector

v = np.linalg.inv(I - gamma * P).dot(r)

print(v)
