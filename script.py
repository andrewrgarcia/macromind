import numpy as np
import scipy
from scipy.optimize import minimize

# Define the model parameters
alpha = 0.33  # Capital share in output
beta = 0.96   # Discount factor
delta = 0.1   # Depreciation rate
rho = 0.9     # Persistence of technology shock
sigma = 0.01  # Standard deviation of technology shock

# Define the production function
def production(k, z):
    return k ** alpha * z ** (1 - alpha)

# Define the utility function
def utility(c):
    return np.log(c)

# Define the Bellman equation
def bellman_equation(k, z, V):
    EV = beta * np.dot(V, [norm.pdf(z_prime, rho*z, sigma) for z_prime in z_grid])
    c = production(k, z) + (1 - delta) * k - V + EV
    return -utility(c)

# Define the state space and shock grid
k_grid = np.linspace(0.1, 10, 100)
z_grid = np.linspace(-3*sigma, 3*sigma, 7)
z_grid[3] = 0  # Set the central point to zero
norm = scipy.stats.norm()

# Solve the model using value function iteration
V = np.zeros((len(k_grid)))
tolerance = 1e-4
max_iterations = 1000
for i in range(max_iterations):
    V_old = V.copy()
    for j, k in enumerate(k_grid):
        for l, z in enumerate(z_grid):
            result = minimize(lambda x: bellman_equation(k, z, x), 0, method='Brent')
            V[j, l] = -result.fun
    if np.max(np.abs(V - V_old)) < tolerance:
        break

# Compute the optimal policy function
g = np.zeros((len(k_grid)))
for j, k in enumerate(k_grid):
    c_values = np.zeros((len(z_grid)))
    for l, z in enumerate(z_grid):
        result = minimize(lambda x: bellman_equation(k, z, x), 0, method='Brent')
        c_values[l] = production(k, z) + (1 - delta) * k - result.x + beta * np.dot(V[:, l], [norm.pdf(z_prime, rho*z, sigma) for z_prime in z_grid])
    g[j] = k_grid[np.argmax(c_values)]

# Plot the results
import matplotlib.pyplot as plt
plt.plot(k_grid, g)
plt.xlabel('Capital')
plt.ylabel('Investment')
plt.title('Optimal investment policy')
plt.show()
