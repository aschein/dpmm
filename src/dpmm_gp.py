"""
Dirichlet Process Mixture Model with Gamma-Poisson Observation Distribution
"""

from numpy import zeros, ones, bincount, mean, exp, log, log2, asarray, ceil, floor, max, min, where, arange, rollaxis, sum
from numpy.random import poisson, gamma, randint, uniform, seed
from random import expovariate
from math_utils import log_sample, sample, vi, logsumexp
from scipy.special import gammaln
# import matplotlib.pyplot as plt

def test_generate_event_data(N, T, alpha):
    """
    Generates N timelines from Dirichlet Process with a Poisson Process observation distribution.
    """
    events_N = [[] for y in range(N)] # interarrival times for each timeline
    
    z_N = zeros(N, dtype=int) # cluster indices
    lambda_N = zeros(N) # cluster parameters (max number of clusters is N)

    for n in xrange(N):
        theta = bincount(z_N)
        theta[0] = alpha # first index is always probability of a new cluster
        [c] = sample(theta) # sample() returns an ndarray of shape (1,)
        if c == 0:
            c = len(theta)
            lambda_N[c - 1] = 2**(c-1) # draw a new cluster parameter
        z_N[n] = c

        # simulate a Poisson process with rate parameter of sampled cluster
        t = 0
        while (t < T):
            t += expovariate(lambda_N[c - 1]) # cluster parameter array doesn't reserve first index
            if (t <= T):
                events_N[n].append(t)

    z_N -= 1 # adjust cluster indices down to match cluster parameter indexing
    return events_N, z_N, lambda_N

def test_generate_count_data(N, T, alpha):
    y_NT = zeros((N, T), dtype=int)

    z_N = zeros(N, dtype=int) # cluster indices
    lambda_N = zeros(N)

    for n in xrange(N):
        theta = bincount(z_N)
        theta[0] = alpha # first index is always probability of a new cluster
        [c] = sample(theta) # sample() returns an ndarray of shape (1,)
        if c == 0:
            c = len(theta)
            lambda_N[c - 1] = 100*c # draw a new cluster parameter
        z_N[n] = c

        for t in xrange(T):
            y_NT[n, t] = lambda_N[c - 1] #+ randint(-1,2)
    z_N -= 1 # adjust cluster indices down to match cluster parameter indexing
    return y_NT, z_N, lambda_N

def printout_test_update(y_NT, z_N):
    avg_y_N = mean(y_NT, axis=1)
    clusters = get_clusters(z_N)
    print [[int(avg_y_N[y]/100) for y in cluster] for cluster in clusters]
    print

def printout_test_update_2(y_N, z_N):
    clusters = get_clusters(z_N)
    print [[int(y_N[y]/100.0) for y in cluster] for cluster in clusters]
    print

def printout_proposal(i, j, z_N, z_prop_N, y_NT):
    avg_y_N = mean(y_NT, axis=1)
    clusters = get_clusters(z_N)
    clusters_prop = get_clusters(z_prop_N)
    zi, zj = z_N[[i, j]]
    if zi == zj:
        print "----SPLIT---"
        [cij] = [x for x in clusters if i in x]
        [csi] = [x for x in clusters_prop if i in x]
        [csj] = [x for x in clusters_prop if j in x]
        print "cluster ij: %s"%str([int(avg_y_N[y]/100) for y in cij])
        print "cluster si: %s"%str([int(avg_y_N[y]/100) for y in csi])
        print "cluster sj: %s"%str([int(avg_y_N[y]/100) for y in csj])
    else:
        print "----MERGE---"
        [ci] = [x for x in clusters if i in x]
        [cj] = [x for x in clusters if j in x]
        [cm] = [x for x in clusters_prop if i in x]
        print "cluster i: %s"%str([int(avg_y_N[y]/100) for y in ci])
        print "cluster j: %s"%str([int(avg_y_N[y]/100) for y in cj])
        print "cluster m: %s"%str([int(avg_y_N[y]/100) for y in cm])
    print

def generate_data(N, T, shape, scale, alpha):
    """
    Generates N timelines from Dirichlet Process with a Poisson Process observation distribution.
    """
    events_N = [[] for y in range(N)] # interarrival times for each timeline
    
    z_N = zeros(N, dtype=int) # cluster indices
    lambda_N = zeros(N) # cluster parameters (max number of clusters is N)

    for n in xrange(N):
        theta = bincount(z_N)
        theta[0] = alpha # first index is always probability of a new cluster
        [c] = sample(theta) # sample() returns an ndarray of shape (1,)
        if c == 0:
            c = len(theta)
            lambda_N[c - 1] = gamma(shape, scale) # draw a new cluster parameter
        z_N[n] = c

        # simulate a Poisson process with rate parameter of sampled cluster
        t = 0
        while (t < T):
            t += expovariate(lambda_N[c - 1]) # cluster parameter array doesn't reserve first index
            if (t <= T):
                events_N[n].append(t)

    z_N -= 1 # adjust cluster indices down to match cluster parameter indexing
    return events_N, z_N, lambda_N

def plot_arrival_data(N, T, events_N):
    for n in xrange(N):
        plt.plot(events_N[n], ones(len(events_N[n]))*n, "o")
        plt.axhline(y=n)
    plt.ylim(-0.5, N-0.5)
    plt.xlim(-0.1, T+0.1)
    plt.show()

def partition_events(events, T):
    """Returns an even partition of continuous interarrival times (events).

    Arguments:
        -- events: list of interarrival times
        -- T:      desired number of discrete time-steps
    """
    interval=ceil(max(events))/float(T)
    return floor([y/interval for y in events]).astype(int)

def get_count_data(events, T):
    return asarray([bincount(partition_events(events, T)) for events in events_N])

def get_clusters(z_N):
    clusters = []
    for idx, c in enumerate(set(z_N)):
        clusters.append([])
        for n in where(z_N == c)[0]:
            clusters[idx].append(n)
    return sorted(clusters, key=lambda x: len(x), reverse=True)

def inference_algorithm_3(y_NT, shape, scale, alpha, num_itns=250, init_z_N=None, true_z_N=None):
    N,T = y_NT.shape

    C = N # maximum number of clusters equals number of observations
    if init_z_N is None:
        z_N = arange(N) # initialize each observation in its own cluster
        N_C = ones(C, dtype=int) # number of observations in each cluster
        y_CT = y_NT.copy() # sum of observations in each cluster

    else:
        z_N = init_z_N.copy()
        N_C = bincount(z_N, minlength=C)
        y_CT = zeros((N,T))
        for c in set(z_N):
            y_CT[c, :] = y_NT[where(z_N==c)[0], :].sum(axis=0)

    active_clusters = set(z_N)
    inactive_clusters = set(range(C)) - active_clusters


    print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
    avg_y_N = mean(y_NT, axis=1)
    true_clusters = get_clusters(true_z_N)
    clusters = get_clusters(z_N)
    print [[int(avg_y_N[y]/100) for y in cluster] for cluster in clusters]
    print

    for itn in xrange(num_itns):
        for n in xrange(N):
            curr_z = z_N[n]

            # decrement sufficient statistics
            N_C[curr_z] -= 1
            y_CT[curr_z, :] -= y_NT[n, :]

            if N_C[curr_z] == 0:
                new_idx = curr_z
                active_clusters.remove(curr_z)
            else:
                new_idx = inactive_clusters.pop()

            log_dist = zeros(len(active_clusters) + 1)
            for k, z in enumerate(active_clusters):
                # P(z_i | z_/i)
                log_dist[k] += log(N_C[z])
                log_dist[k] -= log(N - 1 + alpha)

                # P(y_i | y_/i, z_i)
                log_dist[k] += gammaln(shape + y_NT[n, :].sum() + y_CT[z, :].sum())
                log_dist[k] -= gammaln(shape + y_CT[z, :].sum())
                log_dist[k] += (shape + y_CT[z, :].sum())*log(1.0/scale + N_C[z])
                log_dist[k] -= (shape + y_NT[n, :].sum() + y_CT[z, :].sum())*log(1.0/scale + 1 + N_C[z])
                log_dist[k] -= gammaln(y_NT[n, :].sum() + 1)
            # Analagous steps for new cluster
            log_dist[-1] += log(alpha) 
            log_dist[-1] -= log(N - 1 + alpha)
            log_dist[-1] += gammaln(shape + y_NT[n].sum())
            log_dist[-1] -= gammaln(shape)
            log_dist[-1] += shape*(-log(scale))
            log_dist[-1] -= (shape + y_NT[n].sum())*log(1.0/scale + 1)
            log_dist[-1] -= gammaln(y_NT[n, :].sum() + 1)

            [new_k] = log_sample(log_dist)
            if new_k < len(active_clusters):
                new_z = list(active_clusters)[new_k]
                inactive_clusters.add(new_idx)
            else:
                new_z = new_idx
                active_clusters.add(new_idx)

            z_N[n] = new_z
            N_C[new_z] += 1
            y_CT[new_z, :] += y_NT[n, :]

        if true_z_N is not None:
            if itn%200==0:
                print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
                avg_y_N = mean(y_NT, axis=1)
                true_clusters = get_clusters(true_z_N)
                clusters = get_clusters(z_N)
                print [[int(avg_y_N[y]/100) for y in cluster] for cluster in clusters]
                print

    return z_N

def inference_algorithm_8(y_NT, shape, scale, alpha, M, num_itns=250, init_z_N=None, true_z_N=None):
    N,T = y_NT.shape

    C = N + M - 1 # max num. of clusters is num. observations + num. auxiliary states - 1
    if init_z_N is None:
        z_N = arange(N) # initialize each observation in its own cluster
        N_C = ones(C, dtype=int) # number of observations in each cluster
        y_CT = y_NT.copy() # sum of observations in each cluster

    else:
        z_N = init_z_N.copy()
        N_C = bincount(z_N, minlength=C)
        y_CT = zeros((N,T))
        for c in set(z_N):
            y_CT[c, :] = y_NT[where(z_N==c)[0], :].sum(axis=0)

    lambda_C = gamma(shape, scale, C) # randomly initialize cluster parameters

    active_clusters = set(z_N)
    inactive_clusters = set(range(C)) - active_clusters

    print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
    avg_y_N = mean(y_NT, axis=1)
    true_clusters = get_clusters(true_z_N)
    clusters = get_clusters(z_N)
    print [[int(avg_y_N[y]/100) for y in cluster] for cluster in clusters]
    print

    for itn in xrange(num_itns):
        for n in xrange(N):
            curr_z = z_N[n]

            # decrement sufficient statistics
            N_C[curr_z] -= 1
            y_CT[curr_z, :] -= y_NT[n, :]

            if N_C[curr_z] == 0:
                active_clusters.remove(curr_z)
                indices = list(active_clusters) + [curr_z] + [inactive_clusters.pop() for m in xrange(M - 1)]
                aux_lambda_M = gamma(shape, scale, M-1)
                lambda_C[indices[M-1:]] = aux_lambda_M
            else:
                indices = list(active_clusters) + [inactive_clusters.pop() for m in xrange(M)]
                aux_lambda_M = gamma(shape, scale, M)
                lambda_C[indices[M:]] = aux_lambda_M

            log_dist = zeros(len(indices))
            for k, z in enumerate(indices):
                if z not in active_clusters:
                    log_dist[k] += log(alpha) - log(M)
                    log_dist[k] += gammaln(shape + y_NT[n, :].sum())
                    log_dist[k] -= gammaln(shape)
                    log_dist[k] += shape*(-log(scale))
                    log_dist[k] -= (shape + y_NT[n, :].sum())*log(1.0/scale + 1)
                else:
                    log_dist[k] += log(N_C[z])
                    log_dist[k] += gammaln(shape + y_NT[n, :].sum() + y_CT[z, :].sum())
                    log_dist[k] -= gammaln(shape + y_CT[z, :].sum())
                    log_dist[k] += (shape + y_CT[z, :].sum())*log(1.0/scale + N_C[z])
                    log_dist[k] -= (shape + y_NT[n, :].sum() + y_CT[z, :].sum())*log(1.0/scale + 1 + N_C[z])
                # constants, only included for debugging purposes
                log_dist[k] -= log(N - 1 + alpha)
                log_dist[k] -= gammaln(y_NT[n, :].sum() + 1)

            [new_k] = log_sample(log_dist)
            new_z = indices[new_k]
            if new_z not in active_clusters:
                active_clusters.add(new_z)

            inactive_clusters = inactive_clusters.union(set(indices) - active_clusters)

            z_N[n] = new_z
            N_C[new_z] += 1

            y_CT[new_z, :] += y_NT[n, :]

        if true_z_N is not None:
            if itn%200==0:
                print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
                avg_y_N = mean(y_NT, axis=1)
                true_clusters = get_clusters(true_z_N)
                clusters = get_clusters(z_N)
                print [[int(avg_y_N[y]/100) for y in cluster] for cluster in clusters]
                print
    return z_N

def inference_split_merge(y_NT, shape, scale, alpha, num_itns=250, init_z_N=None, true_z_N=None):
    N, T = y_NT.shape

    C = N # maximum number of clusters equals number of observations
    if init_z_N is None:
        z_N = arange(N) # initialize each observation in its own cluster
        N_C = ones(C, dtype=int) # number of observations in each cluster
        y_CT = y_NT.copy() # sum of observations in each cluster

    else:
        z_N = init_z_N.copy()
        N_C = bincount(z_N, minlength=C)
        y_CT = zeros((N,T))
        for c in set(z_N):
            y_CT[c, :] = y_NT[where(z_N==c)[0], :].sum(axis=0)

    active_clusters = set(z_N)
    inactive_clusters = set(range(C)) - active_clusters

    print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
    avg_y_N = mean(y_NT, axis=1)
    true_clusters = get_clusters(true_z_N)
    clusters = get_clusters(z_N)
    print [[int(avg_y_N[y]/100) for y in cluster] for cluster in clusters]
    print

    def evaluate_split(i, j, z_split_N):
        # print "---SPLIT---"
        # avg_y_N = mean(y_NT, axis=1)
        # clusters = get_clusters(z_N)

        # clusters_split = get_clusters(z_split_N)
        # [cij] = [x for x in clusters if i in x]
        # [csi] = [x for x in clusters_split if i in x]
        # [csj] = [x for x in clusters_split if j in x]
        # print "cluster ij: %s"%str([int(avg_y_N[y]/100) for y in cij])
        # print "cluster si: %s"%str([int(avg_y_N[y]/100) for y in csi])
        # print "cluster sj: %s"%str([int(avg_y_N[y]/100) for y in csj])

        zi, zj = z_N[[i, j]]
        zi_split, zj_split = z_split_N[[i, j]]

        N_split_C = bincount(z_split_N, minlength=C)
        N_zi = N_C[zi]
        N_zi_split = N_split_C[zi_split]
        N_zj_split = N_split_C[zj_split]
        y_zi = y_CT[zi, :].sum()
        y_zi_split = y_NT[where(z_split_N == zi_split)[0], :].sum()
        y_zj_split = y_NT[where(z_split_N == zj_split)[0], :].sum()

        P = log(alpha) + gammaln(N_zi_split) + gammaln(N_zj_split) - gammaln(N_zi)
        Q = - (N_zi_split + N_zj_split - 2)*log(0.5)
        L = 0.0
        L += gammaln(shape + y_zi_split) + gammaln(shape + y_zj_split)
        L -= gammaln(shape) + gammaln(shape + y_zi)
        L += shape*(-log(scale)) + (shape + y_zi)*log(1.0/scale + N_zi)
        L -= (shape + y_zi_split)*log(1.0/scale + N_zi_split) + (shape + y_zj_split)*log(1.0/scale + N_zj_split)
        # print "split: %f"%L

        acc = min([0, P + Q + L])
        return uniform() < exp(acc)

    def evaluate_merge(i, j, z_merge_N):
        # print "----MERGE---"
        # avg_y_N = mean(y_NT, axis=1)
        # clusters = get_clusters(z_N)
        # clusters_merge = get_clusters(z_merge_N)
        # [ci] = [x for x in clusters if i in x]
        # [cj] = [x for x in clusters if j in x]
        # [cm] = [x for x in clusters_merge if i in x]
        # print "cluster i: %s"%str([int(avg_y_N[y]/100) for y in ci])
        # print "cluster j: %s"%str([int(avg_y_N[y]/100) for y in cj])
        # print "cluster m: %s"%str([int(avg_y_N[y]/100) for y in cm])

        zi, zj = z_N[[i, j]]
        zi_merge = z_merge_N[i]

        N_merge_C = bincount(z_merge_N, minlength=C)
        N_zi_merge = N_merge_C[zi_merge]
        N_zi = N_C[zi]
        N_zj = N_C[zj]
        y_zi = y_CT[zi, :].sum()
        y_zj = y_CT[zj, :].sum()
        y_zi_merge = y_NT[where(z_merge_N == zi_merge)[0], :].sum()
        P = -log(alpha) + gammaln(N_zi_merge) - gammaln(N_zi) - gammaln(N_zj)
        Q = (N_zi + N_zj - 2)*log(0.5)
        L = 0.0
        L += gammaln(shape) - shape*log(1.0/scale)
        L += gammaln(shape + y_zi_merge) - (shape + y_zi_merge)*log(1.0/scale + N_zi_merge)
        L += (shape + y_zi)*log(1.0/scale + N_zi) - gammaln(shape + y_zi)
        L += (shape + y_zj)*log(1.0/scale + N_zj) - gammaln(shape + y_zj)
        # print "merge: %f"%L

        acc = min([0, P + Q + L])
        return uniform() < exp(acc)

    for itn in xrange(num_itns):
        if itn%2000==0 or itn==0:
            print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
            printout_test_update(y_NT, z_N)
        # choose a pair of observations uniformly at random
        i = j = 0
        while (i == j):
            i, j = randint(N, size=2)
        zi, zj = z_N[[i, j]]

        # define S to be the set of all observations in i and j's clusters not including i, j
        S = set(list(where(z_N == zi)[0]) + list(where(z_N == zj)[0])) - set([i, j])

        if zi != zj:
            # propose a merge
            z_merge_N = z_N.copy()

            z_merge_N[i] = z_merge_N[j] = zj

            for k in S:
                z_merge_N[k] = zj

            if evaluate_merge(i, j, z_merge_N):
                z_N = z_merge_N.copy()
                active_clusters.remove(zi)
                inactive_clusters.add(zi)
                y_CT[zj, :] += y_CT[zi, :]
                y_CT[zi, :] = 0
                N_C[zj] += N_C[zi]
                N_C[zi] = 0
        else:
            # propose a split
            z_split_N = z_N.copy()

            zi_split = z_split_N[i] = inactive_clusters.pop()
            zj_split = z_split_N[j] = zj 

            for k in S:
                if randint(2):
                    z_split_N[k] = zi_split
                else:
                    z_split_N[k] = zj_split

            if evaluate_split(i, j, z_split_N):
                z_N = z_split_N.copy()
                active_clusters.add(zi_split)
                y_zi_split = y_NT[where(z_split_N == zi_split)[0], :].sum(axis=0)
                y_zj_split = y_NT[where(z_split_N == zj_split)[0], :].sum(axis=0)
                y_CT[zj, :] -= y_zi_split
                y_CT[zi_split, :] = y_zi_split
                N_split_C = bincount(z_split_N, minlength=C)
                N_zi_split = N_split_C[zi_split]
                N_C[zj] -= N_zi_split
                N_C[zi_split] = N_zi_split
            else:
                inactive_clusters.add(zi_split)

        # if true_z_N is not None:
        #     if itn%50000==0:
        #         print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
        #         avg_y_N = mean(y_NT, axis=1)
        #         true_clusters = get_clusters(true_z_N)
        #         clusters = get_clusters(z_N)
        #         print [[int(avg_y_N[y]/100) for y in cluster] for cluster in clusters]
        #         print

    return z_N

def gibbs_scans(S, y_N, z_N, N_C, y_C, shape, scale, num_scans, calc_prob=False, z_result_N=None):
    dist = zeros(y_C.shape[0])
    prod_prob = 0.0
    for k in S:
        zk = z_N[k]
        yk = y_N[k]

        N_C[zk] -= 1
        y_C -= yk

        dist[:] = log(N_C)
        for c in xrange(dist.shape[0]):
            dist[c] += gammaln(shape + yk + y_C[c]) + (shape + y_C[c])*log(1.0/scale + N_C[c])
            dist[c] -= gammaln(shape + y_C[c]) + (shape + yk + y_C[c])*log(1.0/scale + N_C[c] + 1)
        
        if z_result_N is None: # sample a cluster assignment
            z_N[k] = zk = log_sample(dist)
        else: # simulate a sample given the supplied result
            z_N[k] = zk = z_result_N[k]

        N_C[zk] += 1
        y_C[zk] += yk

        if calc_prob:
            prod_prob += dist[zk] - logsumexp(dist)

    return prod_prob

def inference_restricted_split_merge_2(y_NT, shape, scale, alpha, num_itns=250, num_restric_scans=5, num_inter_scans=2, init_z_N=None, true_z_N=None):
    # TODO: This implementation is buggy because the method gibbs_scans() does not support the full scans.
    #       This is because it does not make room for a new group (alpha) and handle 0 counts (after decrementing).
    #       There may well be other bugs.

    N, T = y_NT.shape
    y_N = y_NT.sum(axis=1)

    C = N # maximum number of clusters equals number of observations
    if init_z_N is None:
        z_N = arange(N) # initialize each observation in its own cluster
        N_C = ones(C, dtype=int) # number of observations in each cluster
        y_CT = y_NT.copy() # sum of observations in each cluster
    else:
        z_N = init_z_N.copy()
        N_C = bincount(z_N, minlength=C)
        y_CT = zeros((N,T))
        for c in set(z_N):
            y_CT[c, :] = y_NT[where(z_N==c)[0], :].sum(axis=0)
    
    y_C = y_CT.sum(axis=1)

    active_clusters = set(z_N)
    inactive_clusters = set(range(C)) - active_clusters

    for itn in xrange(num_itns):
        if itn%1000==0 or itn==0:
            print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
            printout_test_update_2(y_N, z_N)

        # choose a pair of observations uniformly at random
        i = j = 0
        while (i == j):
            i, j = randint(N, size=2)
        zi, zj = z_N[[i, j]]

        # define S to be the set of all observations in i and j's clusters not including i, j
        S = set(list(where(z_N == zi)[0]) + list(where(z_N == zj)[0])) - set([i, j])

        z_launch_N = z_N.copy()

        if zi == zj:
            z_launch_N[i] = zi_launch = inactive_clusters.pop()
            z_launch_N[j] = zj_launch = zj
        else:
            zi_launch = zi
            zj_launch = zj

        # select initial launch state randomly
        for k in S:
            if randint(2):
                z_launch_N[k] = zi_launch
            else:
                z_launch_N[k] = zj_launch

        N_launch_C = bincount(z_launch_N, minlength=C)

        # assign zi_launch -> 0, zj_launch -> 1
        N_launch_2 = N_launch_C[[zi_launch, zj_launch]]

        y_launch_2 = zeros(2)
        y_launch_2[0] += y_N[where(z_launch_N == zi_launch)[0]].sum()
        y_launch_2[1] += y_N[where(z_launch_N == zj_launch)[0]].sum()

        # Everyone in zj_launch will be 1, everyone not in zj_launch will be 0.
        # Since we only loop over those in zj_launch OR zi_launch, this implicitly assigns 0 to zi_launch exclusively.
        z_binarized_N = (z_launch_N == zj_launch).astype(int)
        
        # perform M intermediate restricted Gibbs scans
        gibbs_scans(S=S, y_N=y_N, z_N=z_binarized_N, N_C=N_launch_2, y_C=y_launch_2, shape=shape, scale=scale, 
                    num_scans=num_restric_scans, calc_prob=False, z_result_N=None)

        # transfom updated binarized group assignments back to non-binarized
        z_launch_N[list(S)] = (1-z_binarized_N[list(S)])*zi_launch + z_binarized_N[list(S)]*zj_launch

        if zi == zj: # SPLIT
            # z_split_N = z_launch_N.copy()

            # perform one final Gibbs scan from launch state
            prod_prob = gibbs_scans(S=S, y_N=y_N, z_N=z_binarized_N, N_C=N_launch_2, y_C=y_launch_2, shape=shape, scale=scale, 
                                    num_scans=1, calc_prob=True, z_result_N=None)

            # transfom updated binarized group assignments back to non-binarized
            z_split_N[list(S)] = (1-z_binarized_N[list(S)])*zi_launch + z_binarized_N[list(S)]*zj_launch

            # evaluate split
            N_zi = N_C[zi]
            y_zi = y_C[zi]
            N_zi_split, N_zj_split = N_launch_2
            y_zi_split, y_zj_split = y_launch_2

            P = log(alpha) + gammaln(N_zi_split) + gammaln(N_zj_split) - gammaln(N_zi)
            Q = -prod_prob 
            L = shape*log(1.0/scale) - gammaln(shape)
            L += (shape + y_zi)*log(1.0/scale + N_zi) - gammaln(shape + y_zi)
            L += gammaln(shape + y_zi_split) - (shape + y_zi_split)*log(1.0/scale + N_zi_split) 
            L += gammaln(shape + y_zj_split) - (shape + y_zj_split)*log(1.0/scale + N_zj_split)

            acc = min([0, P + Q + L])

            if uniform() < exp(acc):
                z_N = z_split_N.copy()
                active_clusters.add(zi_launch)
                y_C[zj] -= y_launch_2[0]
                y_C[zi_launch] = y_launch_2[0]
                N_C[zj] -= N_launch_2[0]
                N_C[zi_launch] = N_launch_2[0]
            else:
                inactive_clusters.add(zi_launch)

        else: # MERGE
            z_merge_N = z_N.copy()

            # deterministically merge clusters
            zi_merge = z_merge_N[i] = z_merge_N[list(S)] = zj
            N_merge_C = bincount(z_merge_N, minlength=C)

            # binarize the original cluster assignments
            z_result_N = (z_N == zj_launch).astype(int)

            # calculate the hypothetical probability of a Gibbs scan from launch to the original state
            prod_prob = gibbs_scans(S=S, y_N=y_N, z_N=z_binarized_N, N_C=N_launch_2, y_C=y_launch_2, shape=shape, scale=scale, 
                                    num_scans=1, calc_prob=True, z_result_N=z_result_N)
            # evaluate merge
            N_zi, N_zj = N_C[[zi, zj]]
            y_zi, y_zj = y_C[[zi, zj]]
            N_zi_merge = N_merge_C[zi_merge]
            y_zi_merge = y_N[where(z_merge_N == zi_merge)[0]].sum()

            P = -log(alpha) + gammaln(N_zi_merge) - gammaln(N_zi) - gammaln(N_zj)
            Q = prod_prob
            L = gammaln(shape) - shape*log(1.0/scale)
            L += gammaln(shape + y_zi_merge) - (shape + y_zi_merge)*log(1.0/scale + N_zi_merge)
            L += (shape + y_zi)*log(1.0/scale + N_zi) - gammaln(shape + y_zi)
            L += (shape + y_zj)*log(1.0/scale + N_zj) - gammaln(shape + y_zj)

            acc = min([0, P + Q + L])

            if uniform() < exp(acc):
                z_N = z_merge_N.copy()
                active_clusters.remove(zi)
                inactive_clusters.add(zi)
                N_C[zj] += N_C[zi]
                y_C[zj] += y_C[zi]
                N_C[zi] = 0
                y_C[zi] = 0

        # at the end of each iteration, perform some num. of full Gibbs scans
        gibbs_scans(S=range(N), y_N=y_N, z_N=z_N, N_C=N_C, y_C=y_C, shape=shape, scale=scale,
                    num_scans=num_inter_scans, calc_prob=False, z_result_N=None)

    return z_N

def inference_restricted_split_merge(y_NT, shape, scale, alpha, M=5, num_itns=250, init_z_N=None, true_z_N=None):
    N, T = y_NT.shape
    y_N = y_NT.sum(axis=1)

    C = N # maximum number of clusters equals number of observations
    if init_z_N is None:
        z_N = arange(N) # initialize each observation in its own cluster
        N_C = ones(C, dtype=int) # number of observations in each cluster
        y_CT = y_NT.copy() # sum of observations in each cluster

    else:
        z_N = init_z_N.copy()
        N_C = bincount(z_N, minlength=C)
        y_CT = zeros((N,T))
        for c in set(z_N):
            y_CT[c, :] = y_NT[where(z_N==c)[0], :].sum(axis=0)
    
    y_C = y_CT.sum(axis=1)

    active_clusters = set(z_N)
    inactive_clusters = set(range(C)) - active_clusters

    for itn in xrange(num_itns):
        if itn%1000==0 or itn==0:
            print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))
            printout_test_update(y_NT, z_N)
        # choose a pair of observations uniformly at random
        i = j = 0
        while (i == j):
            i, j = randint(N, size=2)
        zi, zj = z_N[[i, j]]

        # define S to be the set of all observations in i and j's clusters not including i, j
        S = set(list(where(z_N == zi)[0]) + list(where(z_N == zj)[0])) - set([i, j])

        z_launch_N = z_N.copy()

        if zi == zj:
            z_launch_N[i] = zi_launch = inactive_clusters.pop()
            z_launch_N[j] = zj_launch = zj
        else:
            zi_launch = zi
            zj_launch = zj

        # select initial launch state randomly
        for k in S:
            if randint(2):
                z_launch_N[k] = zi_launch
            else:
                z_launch_N[k] = zj_launch

        N_launch_C = bincount(z_launch_N, minlength=C)

        # assign zi_launch -> 0, zj_launch -> 1
        N_launch_2 = zeros(2)
        N_launch_2[0] = N_launch_C[zi_launch]
        N_launch_2[1] = N_launch_C[zj_launch]

        y_launch_2T = zeros((2, T))
        y_launch_2T[0, :] += y_NT[where(z_launch_N == zi_launch)[0], :].sum(axis=0)
        y_launch_2T[1, :] += y_NT[where(z_launch_N == zj_launch)[0], :].sum(axis=0)

        # perform M intermediate restricted Gibbs scans
        dist = zeros(2)
        for m in xrange(M):
            for k in S:
                yk_T = y_NT[k, :]
                yk = yk_T.sum()
                zk_launch = z_launch_N[k]
                N_launch_2[int(zk_launch == zj_launch)] -= 1
                y_launch_2T[int(zk_launch == zj_launch), :] -= yk_T

                y_launch_0, y_launch_1 = y_launch_2T.sum(axis=1)

                dist = log(N_launch_2)
                dist[0] += gammaln(shape + yk + y_launch_0) + (shape + y_launch_0)*log(1.0/scale + N_launch_2[0])
                dist[0] -= gammaln(shape + y_launch_0) + (shape + yk + y_launch_0)*log(1.0/scale + N_launch_2[0] + 1)

                dist[1] += gammaln(shape + yk + y_launch_1) + (shape + y_launch_1)*log(1.0/scale + N_launch_2[1])
                dist[1] -= gammaln(shape + y_launch_1) + (shape + yk + y_launch_1)*log(1.0/scale + N_launch_2[1] + 1)
                [t] = log_sample(dist)

                zk_launch = z_launch_N[k] = zi_launch if t == 0 else zj_launch
                N_launch_2[int(zk_launch == zj_launch)] += 1
                y_launch_2T[int(zk_launch == zj_launch), :] += yk_T

        if zi == zj: # SPLIT
            # perform one final Gibbs scan from launch state
            Q = 0.0 # maintain product of probabilities
            for k in S:
                yk_T = y_NT[k, :]
                yk = yk_T.sum()
                zk_launch = z_launch_N[k]
                N_launch_2[int(zk_launch == zj_launch)] -= 1
                y_launch_2T[int(zk_launch == zj_launch), :] -= yk_T
                y_launch_0, y_launch_1 = y_launch_2T.sum(axis=1)

                dist = log(N_launch_2)
                dist[0] += gammaln(shape + yk + y_launch_0) + (shape + y_launch_0)*log(1.0/scale + N_launch_2[0])
                dist[0] -= gammaln(shape + y_launch_0) + (shape + yk + y_launch_0)*log(1.0/scale + 1 + N_launch_2[0])
                dist[1] += gammaln(shape + yk + y_launch_1) + (shape + y_launch_1)*log(1.0/scale + N_launch_2[1])
                dist[1] -= gammaln(shape + y_launch_1) + (shape + yk + y_launch_1)*log(1.0/scale + 1 + N_launch_2[1])
                norm_const = logsumexp(dist)

                [t] = log_sample(dist)
                zk_launch = z_launch_N[k] = zi_launch if t == 0 else zj_launch
                N_launch_2[int(zk_launch == zj_launch)] += 1
                y_launch_2T[int(zk_launch == zj_launch), :] += yk_T

                Q -= dist[t] - norm_const

            # evaluate split
            N_zi = N_C[zi]
            y_zi = y_CT[zi, :].sum()
            N_zi_split, N_zj_split = N_launch_2
            y_zi_split, y_zj_split = y_launch_2T.sum(axis=1)
            P = log(alpha) + gammaln(N_zi_split) + gammaln(N_zj_split) - gammaln(N_zi)
            L = 0.0
            L += gammaln(shape + y_zi_split) + gammaln(shape + y_zj_split)
            L -= gammaln(shape) + gammaln(shape + y_zi)
            L += shape*(-log(scale)) + (shape + y_zi)*log(1.0/scale + N_zi)
            L -= (shape + y_zi_split)*log(1.0/scale + N_zi_split) + (shape + y_zj_split)*log(1.0/scale + N_zj_split)

            # printout_proposal(i, j, z_N, z_launch_N, y_NT)
            # print "P: %f\nQ: %f\nL: %f"%(P, Q, L)

            acc = min([0, P + Q + L])

            if uniform() < exp(acc):
                z_N = z_launch_N.copy()
                active_clusters.add(zi_launch)
                y_CT[zj, :] -= y_launch_2T[0, :]
                y_CT[zi_launch, :] = y_launch_2T[0, :]
                N_C[zj] -= N_launch_2[0]
                N_C[zi_launch] = N_launch_2[0]
            else:
                inactive_clusters.add(zi_launch)

        else: # MERGE
            z_merge_N = z_N.copy()
            zi_merge = z_merge_N[i] = z_merge_N[j] = zj
            for k in S:
                z_merge_N[k] = zj

            # calculate the hypothetical probability of a Gibbs scan from launch to the original state
            Q = 0.0
            for k in S:
                yk_T = y_NT[k, :]
                yk = yk_T.sum()
                zk = z_N[k]
                assert ((zk == zi_launch) or (zk == zj_launch))
                zk_launch = z_launch_N[k]
                N_launch_2[int(zk_launch == zj_launch)] -= 1
                y_launch_2T[int(zk_launch == zj_launch), :] -= yk_T
                y_launch_0, y_launch_1 = y_launch_2T.sum(axis=1)

                dist = log(N_launch_2)
                dist[0] += gammaln(shape + yk + y_launch_0) + (shape + y_launch_0)*log(1.0/scale + N_launch_2[0])
                dist[0] -= gammaln(shape + y_launch_0) + (shape + yk + y_launch_0)*log(1.0/scale + N_launch_2[0] + 1)
                dist[1] += gammaln(shape + yk + y_launch_1) + (shape + y_launch_1)*log(1.0/scale + N_launch_2[1])
                dist[1] -= gammaln(shape + y_launch_1) + (shape + yk + y_launch_1)*log(1.0/scale + N_launch_2[1] + 1)
                norm_const = logsumexp(dist)

                t = int(zk == zj_launch)
                z_launch_N[k] = zk
                N_launch_2[t] += 1
                y_launch_2T[t] += yk_T

                Q += dist[t] - norm_const
            
            # evaluate merge
            N_merge_C = bincount(z_merge_N, minlength=C)
            N_zi_merge = N_merge_C[zi_merge]
            N_zi = N_C[zi]
            N_zj = N_C[zj]
            y_zi = y_CT[zi, :].sum()
            y_zj = y_CT[zj, :].sum()
            y_zi_merge = y_NT[where(z_merge_N == zi_merge)[0], :].sum()

            P = -log(alpha) + gammaln(N_zi_merge) - gammaln(N_zi) - gammaln(N_zj)
            L = 0.0
            L += gammaln(shape) - shape*log(1.0/scale)
            L += gammaln(shape + y_zi_merge) - (shape + y_zi_merge)*log(1.0/scale + N_zi_merge)
            L += (shape + y_zi)*log(1.0/scale + N_zi) - gammaln(shape + y_zi)
            L += (shape + y_zj)*log(1.0/scale + N_zj) - gammaln(shape + y_zj)
            
            if L < -1e35:
                pass

            # printout_proposal(i, j, z_N, z_merge_N, y_NT)
            # print "P: %f\nQ: %f\nL: %f"%(P, Q, L)

            acc = min([0, P + Q + L])
            if uniform() < exp(acc):
                z_N = z_merge_N.copy()
                active_clusters.remove(zi)
                inactive_clusters.add(zi)
                y_CT[zj, :] += y_CT[zi, :]
                y_CT[zi, :] = 0 
                N_C[zj] += N_C[zi]
                N_C[zi] = 0
    return z_N

if __name__ == "__main__":
    N = 50
    T = 10
    shape, scale = 20, 20
    alpha = 5

    # events_N, true_z_N, true_lambda_N = generate_event_data(N, T, shape, scale, alpha)
    # events_N, true_z_N, true_lambda_N = test_generate_event_data(N, T, alpha)
    # plot_arrival_data(N, T, events_N)

    seed(10)
    # seed(20)
    y_NT, true_z_N, true_lambda_N = test_generate_count_data(N, T, alpha)
    seed(randint(0,1000))
    alpha = 10

    num_itns = 4000
    # init_z_N = zeros(y_NT.shape[0], dtype=int) # all start in one group
    init_z_N = arange(N)

    # print "-----ALGORITHM 8-----"
    # z_N = inference_algorithm_8(y_NT, shape, scale, alpha, M=10, num_itns=num_itns, init_z_N=init_z_N, true_z_N=true_z_N)
    print "-----ALGORITHM 3-----"
    z_N = inference_algorithm_3(y_NT, shape, scale, alpha, num_itns=num_itns, init_z_N=init_z_N, true_z_N=true_z_N)
    # print "-----Split-Merge-----"
    # z_N = inference_split_merge(y_NT, shape, scale, alpha, num_itns=2001*10, init_z_N=init_z_N, true_z_N=true_z_N)
    # print "----Restricted SM----"
    # z_N = inference_restricted_split_merge(y_NT, shape, scale, alpha, M=5, num_itns=6001, init_z_N=init_z_N, true_z_N=true_z_N)
    # for x in xrange(50):
    #     print "-----ALGORITHM 3-----"
    #     z_N = inference_algorithm_3(y_NT, shape, scale, alpha, num_itns=50, init_z_N=z_N, true_z_N=true_z_N)
    #     print "----Restricted SM----"
        # z_N = inference_restricted_split_merge(y_NT, shape, scale, alpha, M=5, num_itns=50, init_z_N=z_N, true_z_N=true_z_N)
    # print "----Restricted SM----"
    # z_N = inference_restricted_split_merge_2(y_NT, shape, scale, alpha, num_itns=6001, num_restric_scans=5, num_inter_scans=2, init_z_N=init_z_N, true_z_N=true_z_N)
