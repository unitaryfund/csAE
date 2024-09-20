import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from signals import *
from frequencyestimator import *
import itertools
import math

sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
sns.set_context("poster", font_scale = .45, rc={"grid.linewidth": 0.8})

# For reproducibility
np.random.seed(8)
# Set the per oracle noise parameter (See Eq. 18)
eta = 0
# Set the array parameters (See Thm. II.2 and Eq. 12)
narray = [3, 3, 3, 3, 2, 2, 2, 2]
narray = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
q = 6
narray = [2] * q
narray = [3,3, 2, 2, 2, 2]
# Set the actual amplitude
a = 0.8
theta = np.arcsin(a)

# This sets up the simulation that simulates the measured amplitudes at the various physical locations.
# It uses a C=1.5 value, which corresponds to the sampling schedule given in Eq. 16. The variable C here
# is the parameter K in the paper.
ula_signal = TwoqULASignal(M=narray, C=3)

# Number of Monte Carlo trials used to estimate statistics. We tend to use 500 in the paper. Choose 100 here for speed.
num_mc = 10
thetas = np.zeros(num_mc, dtype=float)
errors = np.zeros(num_mc, dtype=float)

# Sets up the ESPIRIT object to estimate the amplitude
espirit = ESPIRIT()

signs = [1] * (q + 1)
signs[0] = 1
sign_overlap = 0
# signs = [1, 1, 1, -1, 1, 1]
# signs = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0]

signal = ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta)
all_signs = [s for s in itertools.product([1.0, -1.0], repeat=len(signal)-1)]

for k in range(num_mc):
    # This estimates the covariance matrix of Eq. 8 using the approch given in DOI:10.1109/LSP.2015.2409153

    ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta)
    objective = -np.inf

    for signs in all_signs:
        signs = [1.0] + list(signs)
        # print(signs)
        # signal = ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta, signs=signs)
        signal = ula_signal.update_signal_signs(signs)
        R = ula_signal.get_cov_matrix_toeplitz(signal)
        # This estimates the angle using the ESPIRIT algorithm
        theta_est, eigs = espirit.estimate_theta_toeplitz(R)
        objective_new = np.abs(espirit.eigs[0]) - np.abs(espirit.eigs[1])
        # print(f'Objective: {objective}')
        if math.isclose(np.abs(np.dot(ula_signal.signs_exact, signs))/len(signs), 1):
            # print('CORRECT ANSWER')
            correct_objective = objective_new
        # print(f'Objective New: {objective_new}')
        # print(f'Signs Exact: {ula_signal.signs_exact}')
        # print(f'Signs:       {signs}\n')
        # print(f'Error: {np.abs(np.sin(theta)-np.sin(theta_est))}')

        if objective_new > objective:
            # print('here')
            # print(f'objective: {objective}')
            # print(f'objective_new: {objective_new}')
            objective = objective_new
            signs_found = signs

            # Estimate the error between estimated a and actual a
            error = np.abs(np.sin(theta) - np.sin(theta_est))
            thetas[k] = theta_est
            errors[k] = error
            # print(theta_est)
        # else:
        #     signs[j + 1] = -1 * signs[j + 1]
        #     signs_found = signs

    # print('\n')

    print(f'Objective Found: {objective}')
    print(f'Correct Objective: {correct_objective}')
    print(f'Signs found: {signs_found}\n')
    # print(f'Signs exact: {ula_signal.signs_exact}')
    sign_overlap = sign_overlap + np.abs(np.dot(ula_signal.signs_exact, signs_found)) / len(signs_found) / num_mc

    # R = ula_signal.get_cov_matrix(theta, n_samples=ula_signal.n_samples, eta=eta)
    # theta_est = espirit.estimate_theta(R)

# Compute the total number of queries. The additional count of ula_signal.n_samples[0]/2 is to
# account for the fact that the Grover oracle has two invocations of the unitary U, but is
# preceded by a single invocation of U (see Eq. 2 in paper). This accounts for the shots required
# for that single U operator, which costs half as much as the Grover oracle.
num_queries = np.sum(np.array(ula_signal.depths) * np.array(ula_signal.n_samples)) + ula_signal.n_samples[0] / 2
# Compute the maximum single query
max_single_query = np.max(ula_signal.depths)

print(f'Average sign overlap: {sign_overlap}')
print(f'Array parameters: {narray}')
print(f'Number of queries: {num_queries}')
print(f'theta: {theta}')
print(f'Ave theta estimated: {np.mean(thetas)}')
print(f'a = {a}; a_est = {np.sin(np.mean(thetas))}')
print(f'Max Single Query: {max_single_query}')
print(f'99% percentile: {np.percentile(errors, 99):e}')
print(f'95% percentile: {np.percentile(errors, 95):e}')
print(f'68% percentile: {np.percentile(errors, 68):e}')
print(f'99% percentile constant: {np.percentile(errors, 99) * num_queries:f}')
print(f'95% percentile constant: {np.percentile(errors, 95) * num_queries:f}')
print(f'68% percentile constant: {np.percentile(errors, 68) * num_queries:f}')
print()

signal = ula_signal.update_signal_signs(ula_signal.signs_exact)
ula_signal_exact = ula_signal.get_ula_signal(signal)
signal = ula_signal.update_signal_signs(signs_found)
ula_signal_found = ula_signal.get_ula_signal(signal)
# signal = ula_signal.update_signal_signs([1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
# ula_signal_bad = ula_signal.get_ula_signal(signal)
# plt.plot(np.real(ula_signal_exact))
# plt.plot(np.real(ula_signal_found))
# plt.plot(np.real(ula_signal_bad))
# plt.show()