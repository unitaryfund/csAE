'''
This code is from https://github.com/qiskit-community/ChebAE/blob/main/chebae.ipynb with some small modifications.
The chebae() function records the maximum depth as well as the time complexity and returns those.
So no changes to the actual algorithm, just returning extra information from the algorithm.
'''



import numpy as np
from scipy.special import eval_chebyt as cheb
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import json


def invert(a_int,deg,p):
    
    # Leverage T_d(a) = cos( d arccos(a) )
    # or |T_d(a)|^2 = cos^2( d arccos(a) )
    #     = 0.5 + 0.5 * cos( 2d arccos(a) )
    # by considering theta = arccos(a).
    theta_int = np.arccos(a_int)

    # Now each 'half period', which is an interval
    # of the form [ t*pi/2d,  (t+1)*pi/2d ] for integer k,
    # has cos( 2d theta ) strictly increasing/decreasing.
    
    c = np.pi/(2*deg)
    # intervals have form [ t*c,  (t+1)*c ]

    # find out which half-period we are on.
    t = np.floor(theta_int/c)
    theta_lo = c*t
    theta_hi = theta_lo + c
    # now [theta_lo, theta_hi] bound a section of
    # cos^2(deg*theta) where the slope never changes sign
    # if t is even then we are decreasing.
    # if t is odd we are increasing

    # cos(deg*theta)^2 = p
    # cos(2*deg*theta) = 2*p - 1
    # has two solutions. pick solution near 0 or 2*c
    # depending on which half of the period we are on
    if t % 2 == 0:
        theta = np.arccos(2*p-1)/(2*deg)
    else:
        theta = 2*c - np.arccos(2*p-1)/(2*deg)

    # shift solution into correct period
    k = t//2
    theta += np.pi*k/deg

    return np.cos(theta)



# find a deg > min_deg such that cheb(deg, a)**2 has no
# extrema for any a in [a_min,a_max]
def find_next_cheb(a_min, a_max, min_deg=0, odd=False):

    # Step 1: convert to theta.
    theta_lo = np.arccos(a_max)
    theta_hi = np.arccos(a_min)
    
    # Step 2: get highest possible degree.
    deg = int((np.pi/2)/(theta_hi-theta_lo))
    if odd and deg % 2 == 0: deg += 1

    # Step 3: search for the highest degree without any extrema.
    while deg > min_deg:
        
        if int(2*deg*theta_lo/np.pi) == int(2*deg*theta_hi/np.pi):
            # Done!
            return deg
        
        if odd: deg -= 2
        else: deg -= 1
            
    return None # Couldn't find a degree > min_deg.



# Say we tossed a coin with unknown bias Nshots many times,
# and we want a confidence interval with confidence >= 1-delta.
# What is the widest that this interval could be?
# Relies on Clopper-Pearson confidence interval method.
def max_error_cp(delta, Nshots):
    max_error = 0
    
    # Loop over all possible numbers of heads.
    for counts in range(0,Nshots+1):
        lower,upper = proportion_confint(counts, Nshots,
                                         method="beta",
                                         alpha=delta)
        if (upper-lower)/2 > max_error:
            max_error = (upper-lower)/2
    
    return max_error
        
# IQAE's computation of the maximum error on theta.
# This seems to loop over many possible probabilities x.
# We could have optimized this, but we want to remain
# faithful to IQAE's implementation. 
# But we did add a cache to make it evaluate faster.
max_error_iqae_cache = {}
def max_error_iqae(delta_T, N_shots):
    if str([delta_T,N_shots]) in max_error_iqae_cache:
        return max_error_iqae_cache[str([delta_T,N_shots])]
    
    g_CP_y = []
    for x in np.linspace(0, 1, 10000):
        p_min, p_max = proportion_confint(int(x*N_shots), N_shots, method='beta', alpha=delta_T)
        g_CP_y.append(np.abs(np.arccos(1-2*p_max)-np.arccos(1-2*p_min))/2)
    out = np.max(g_CP_y)
    
    max_error_iqae_cache[str([delta_T,N_shots])] = out
    
    return out


def chebae(a_true, eps, delta,
           nu=8, r=2, Nshots=100,
           IQAE_cutoff=False, odd=False):

    # Step 1: determine the total number of confidence intervals
    # and distribute failure probability budget evenly among them
    T = max(1,int(np.ceil(np.log(1/(2*eps))/np.log(r))))
    delta_T = delta/T 

    
    # Step 2: precompute cutoff parameters.
    if IQAE_cutoff: L_max = max_error_iqae(delta_T, Nshots)
    else: err_max = max_error_cp(delta_T, Nshots)

        
    # Step 3: Initialize
    a_min, a_max = 0, 1 # confidence interval
    num_flips, num_heads = 0, 0 # coin toss tally    
    deg = 1 # degree
    queries = 0 # only count queries to Pi oracle
    max_depth = 0
    time_complexity = 0

    
    # Step 4:
    while a_max - a_min > eps*2:

        # Step 4(a)
        # Try to find a better polynomial with degree > r*deg.
        new_deg = find_next_cheb(a_min, a_max,
                                 min_deg=deg*r,
                                 odd=odd)

        # Found a better polynomial? If so, reset the counts.
        if new_deg is not None:
            deg = new_deg
            num_flips, num_heads = 0, 0
            
        #add deg to time complexity
        time_complexity += int(np.floor(deg/2))

        #check if deg is larger than max_depth
        if int(np.floor(deg/2)) > max_depth:
                max_depth = int(np.floor(deg/2))

        # Step 4(b): determine 'late' or 'early' to avoid taking too many samples
        # by setting N_shots_i - the number of shots in this iteration
        if IQAE_cutoff:
            # follow IQAE's implementation to decide when to reduce N_shots_i
            # deg = 2*k+1
            # K_i = 4*k+2 = 2*deg
            K_i = 2*deg
            eps_theta = eps / np.pi
            if K_i > int(L_max/eps_theta):
                Nshots_i = int((L_max/eps_theta)*Nshots/K_i/10)
                if Nshots_i == 0: Nshots_i = 1
            else:
                Nshots_i = Nshots
        else:
            gap = cheb(deg, a_max)**2 - cheb(deg, a_min)**2
            if err_max * (a_max - a_min)/gap < nu*eps:
                Nshots_i = 1      # late: sample one-at-a-time
            else:
                Nshots_i = Nshots # early: take lots of samples

        # Step 4(c): Simulate the quantum computer to toss coins
        prob = cheb(deg, a_true)**2 
        for i in range(Nshots_i):
            if np.random.random() < prob: num_heads += 1
            num_flips += 1
            # final measurement doesn't count as a query,
            # to be consistent with 1912.05559's benchmarking
            queries += int(np.floor(deg/2))
            
        # Step 4(d): determine confidence interval for prob
        p_min, p_max = proportion_confint(num_heads, num_flips,
                                          method="beta",
                                          alpha=delta_T)
        
        # Step 4(e): back-propagate [p_min,p_max] to confidence
        # interval for [a_min_star, a_max_star] for a_true
        a_int = np.mean([a_min,a_max])
        a_min_star = invert(a_int,deg,p_min)
        a_max_star = invert(a_int,deg,p_max)
        a_min_star, a_max_star = sorted([a_min_star, a_max_star])

        # prevent floating point glitches
        a_min_star -= 1e-15
        a_max_star += 1e-15
        
        # Step 5(d): updat ethe interval
        a_min, a_max = max(a_min, a_min_star), min(a_max, a_max_star)

    # Status: did was a_true indeed in the confidence interval?
    # In other words, was the estimate actually accurate?
    status = bool((a_min <= a_true) and (a_true <= a_max))


    return {'queries': queries,
            'max_depth': max_depth,
            'time_complexity': time_complexity,
            'a_min': a_min,
            'a_max': a_max,
            'a_hat': np.mean([a_min,a_max]),
            'status': status}