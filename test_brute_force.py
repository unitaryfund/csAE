import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from signals import *
from frequencyestimator import *
import itertools
import math
from scipy.stats import binom

sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
sns.set_context("poster", font_scale = .45, rc={"grid.linewidth": 0.8})

# For reproducibility
#22, 26
np.random.seed(14)
# Set the per oracle noise parameter (See Eq. 18)
eta = 0
# Set the array parameters (See Thm. II.2 and Eq. 12)
narray = [3, 3, 3, 3, 2, 2, 2, 2]
narray = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
q = 6
narray = [2] * q
narray = [2, 2, 2, 2, 2, 2]
narray = [2, 2, 2, 2, 2, 3]
# Set the actual amplitude
a = 0.2
theta = np.arcsin(a)

# This sets up the simulation that simulates the measured amplitudes at the various physical locations.
# It uses a C=1.5 value, which corresponds to the sampling schedule given in Eq. 16. The variable C here
# is the parameter K in the paper.
ula_signal = TwoqULASignal(M=narray, C=5.0)

# Number of Monte Carlo trials used to estimate statistics. We tend to use 500 in the paper. Choose 100 here for speed.
num_mc = 1
thetas = np.zeros(num_mc, dtype=float)
errors = np.zeros(num_mc, dtype=float)

# Sets up the ESPIRIT object to estimate the amplitude
espirit = ESPIRIT()

sign_overlap = 0

signal = ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta)
all_signs = [s for s in itertools.product([1.0, -1.0], repeat=len(signal)-1)]

# all_signs = [[1.0]*(len(signal)-1)]
# all_signs = [ula_signal.signs_exact[1:]]

for k in range(num_mc):
    print(f'Trial {k+1} of {num_mc}')
    # This estimates the covariance matrix of Eq. 8 using the approch given in DOI:10.1109/LSP.2015.2409153

    ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta)
    objective = -np.inf

    for signs in all_signs:
    # for signs in [ula_signal.signs_exact[1:]]:
        signs = [1.0] + list(signs)
        # print(signs)
        # signal = ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta, signs=signs)
        signal = ula_signal.update_signal_signs(signs)
        R = ula_signal.get_cov_matrix_toeplitz(signal)
        # This estimates the angle using the ESPIRIT algorithm
        theta_est, eigs = espirit.estimate_theta_toeplitz(R, s0=np.real(signal[0])**2)


        objective_new = np.abs(espirit.eigs[0]) - np.abs(espirit.eigs[1])
        # print(f'Objective: {objective}')
        if math.isclose(np.linalg.norm(np.array(ula_signal.signs_exact) - np.array(signs)), 0):
            print(f'     CORRECT ANSWER')
            print(f'     angle {-np.angle(eigs) / np.pi / 4}; objective {objective_new}')
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
            print(f'angle {-np.angle(eigs) / np.pi / 4}; objective {objective}')
            # print(np.real(signal[0])**2)
            # print(np.abs(np.imag(signal[0])))
            # print(np.cos(2*theta_est)**2)
            # print(np.abs(np.sin(2*theta_est)))
            # if np.abs(np.real(signal[0])**2 - np.cos(2*theta_est)**2) > 0.25:
            #     print('here')
            #     theta_est = np.abs(np.pi / 2.0 - np.abs(theta_est))


            # Estimate the error between estimated a and actual a
            error = np.abs(np.sin(theta) - np.sin(theta_est))
            thetas[k] = theta_est

    p_exact = np.cos((2*ula_signal.depths+1)*(theta))**2
    p_o2 = np.cos((2 * ula_signal.depths + 1) * (thetas[k]/2.0)) ** 2
    p_o4 = np.cos((2 * ula_signal.depths + 1) * (thetas[k] / 4.0)) ** 2
    p_same = np.cos((2*ula_signal.depths+1)*(thetas[k]))**2
    p_s2 = np.cos((2 * ula_signal.depths + 1) * (np.pi/2-thetas[k])) ** 2
    p_s4 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - thetas[k])) ** 2
    p_s2_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 2 - thetas[k]/2)) ** 2
    p_s4_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - thetas[k]/2)) ** 2
    # obj_o2 = np.linalg.norm(ula_signal.measurements - np.cos((2 * ula_signal.depths + 1) * (thetas[k]/2.0)) ** 2)
    # obj_same = np.linalg.norm(ula_signal.measurements - np.cos((2*ula_signal.depths+1)*(thetas[k]))**2)
    # obj_s2  = np.linalg.norm(ula_signal.measurements - np.cos((2 * ula_signal.depths + 1) * (np.pi/2-thetas[k])) ** 2)
    # obj_s4  = np.linalg.norm(ula_signal.measurements - np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - thetas[k])) ** 2)
    # obj_s2_o2 = np.linalg.norm(ula_signal.measurements - np.cos((2 * ula_signal.depths + 1) * (np.pi / 2 - thetas[k]/2)) ** 2)
    # obj_s4_o2 = np.linalg.norm(ula_signal.measurements - np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - thetas[k]/2)) ** 2)
    l_exact = np.sum(
        np.log([1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk],
                                  p_exact[kk]) for kk in
                range(len(ula_signal.n_samples))]))

    l_o2 = np.sum(
        np.log([1e-75+binom.pmf(ula_signal.n_samples[kk]*ula_signal.measurements[kk], ula_signal.n_samples[kk], p_o2[kk]) for kk in
         range(len(ula_signal.n_samples))]))
    l_o4 = np.sum(
        np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_o4[kk]) for kk in
         range(len(ula_signal.n_samples))]))
    l_same = np.sum(
        np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_same[kk]) for kk in
         range(len(ula_signal.n_samples))]))
    l_s2 = np.sum(
        np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s2[kk]) for kk in
         range(len(ula_signal.n_samples))]))
    l_s4 = np.sum(
        np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s4[kk]) for kk in
         range(len(ula_signal.n_samples))]))
    l_s2_o2 = np.sum(
        np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s2_o2[kk]) for kk in
         range(len(ula_signal.n_samples))]))
    l_s4_o2 = np.sum(
        np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s4_o2[kk]) for kk in
         range(len(ula_signal.n_samples))]))
    print(f'theta_exact obj:           {l_exact}\n')

    print(f'2*theta_found obj:         {l_same}')
    print(f'2*theta found:             {2*thetas[k] / np.pi}\n')

    print(f'pi-2*theta_found obj:      {l_s2}')
    print(f'pi-2*theta found:          {1.0-2 * thetas[k] / np.pi}\n')

    print(f'pi/2-2*theta_found obj:    {l_s4}')
    print(f'pi/2-2*theta found:        {0.5 - 2 * thetas[k] / np.pi}\n')

    print(f'theta_found obj:           {l_o2}')
    print(f'theta found:               {thetas[k] / np.pi}\n')

    print(f'theta_found/2 obj:         {l_o4}')
    print(f'theta found:               {thetas[k] / 2.0 / np.pi}\n')

    print(f'pi - theta_found obj:    {l_s2_o2}')
    print(f'pi - theta found:        {1.0 - thetas[k] / np.pi}\n')

    print(f'pi/2 - theta_found obj:    {l_s4_o2}')
    print(f'pi/2 - theta found:        {0.5 - thetas[k] / np.pi}\n')

    print('FINAL ANGLE FOUND')
    which_correction = np.argmax([l_same, l_s2, l_s4, l_o2, l_o4, l_s2_o2, l_s4_o2])
    if which_correction == 1:
        thetas[k] = np.pi/2.0 - thetas[k]
    elif which_correction == 2:
        thetas[k] = np.pi/4.0 - thetas[k]
    elif which_correction == 3:
        thetas[k] = 0.5*thetas[k]
    elif which_correction == 4:
        thetas[k] = 0.25*thetas[k]
    elif which_correction == 5:
        thetas[k] = np.pi / 2.0 - 0.5 * thetas[k]
    elif which_correction == 6:
        thetas[k] = np.pi / 4.0 - 0.5 * thetas[k]

    print(f'2*theta corrected:         {2*thetas[k] / np.pi}')
    print(f'2*theta exact:             {2*theta / np.pi}')
    print(f'Objective Found: {objective}')
    # print(f'Correct Objective: {correct_objective}')
    print(f'Signs found: {signs_found}')
    print(f'Signs exact: {ula_signal.signs_exact}\n')
    sign_overlap = sign_overlap + np.abs(np.dot(ula_signal.signs_exact, signs_found)) / len(signs_found) / num_mc


    errors[k] = np.abs(np.abs(np.sin(theta)) - np.abs(np.sin(thetas[k])))

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
# plt.plot(np.real(ula_signal.measurements))
# plt.plot(np.cos((2*ula_signal.depths+1)*(thetas[k]))**2)
# plt.plot(np.cos((2*ula_signal.depths+1)*(np.pi/2-thetas[k]))**2)
# plt.plot(np.cos((2*ula_signal.depths+1)*(np.pi/4-thetas[k]))**2)
# plt.show()