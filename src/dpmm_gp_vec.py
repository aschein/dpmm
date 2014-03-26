from numpy import zeros, ones, bincount, mean, exp, log, log2, asarray, ceil, floor, max, min, where, arange, rollaxis, sum
from numpy.random import poisson, gamma, randint, uniform, seed
from random import expovariate
from math_utils import log_sample, sample, vi, logsumexp
from scipy.special import gammaln

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('error')

def generate_data(N, T, shape, scale, alpha):
    y_NT = zeros((N, T), dtype=int)

    z_N = zeros(N, dtype=int)
    delta_NT = zeros((N, T))

    for n in xrange(N):
        theta = bincount(z_N)
        theta[0] = alpha
        [c] = sample(theta)
        if c == 0:
            c = len(theta)
            delta_NT[c-1, :] = gamma(shape, scale, T)
        z_N[n] = c

        y_NT[n, :] = poisson(delta_NT[c-1, :])

    z_N -= 1
    return y_NT, z_N, delta_NT

def generate_test_data(N, T, shape, scale, alpha):
    y_NT = zeros((N, T), dtype=int)

    z_N = zeros(N, dtype=int)
    delta_NT = zeros((N, T))

    for n in xrange(N):
        theta = bincount(z_N)
        theta[0] = alpha
        [c] = sample(theta)
        if c == 0:
            c = len(theta)
            delta_NT[c-1, :] = gamma(shape, scale, T)
        z_N[n] = c

        y_NT[n, :] = delta_NT[c-1, :]

    z_N -= 1
    return y_NT, z_N, delta_NT

def generate_easy_data():
    y_NT = np.zeros(())

def get_clusters(z_N):
    clusters = []
    for c in set(z_N):
        clusters.append((where(z_N==c)[0]))
    clusters.sort(key=lambda x: len(x), reverse=True)
    return clusters

def print_proposal(i, j, z_N, z_prop_N, y_NT):
    zi, zj = z_N[[i, j]]
    zi_prop, zj_prop = z_prop_N[[i,j]]

    # clusters = [y_NT[c, :] for c in get_clusters(z_N)]
    # clusters_prop = [y_NT[c, :] for c in get_clusters(z_prop_N)]
    clusters = get_clusters(z_N)
    clusters_prop = get_clusters(z_prop_N) 

    if zi == zj:
        print "----SPLIT----\n"
        # print "i: %s\n"%(str(y_NT[i, :]))
        # print "j: %s\n"%(str(y_NT[j, :]))

        print "(i, j) : (%d, %d)"%(i, j)
        map(lambda x: i in x, clusters)
        [cij] = [x for x in clusters if i in x]
        [csi] = [x for x in clusters_prop if i in x]
        [csj] = [x for x in clusters_prop if j in x]
        print "Current: \n%s"%(str(y_NT[cij, :]))
        print "Split 1: \n%s"%(str(y_NT[csi, :]))
        print "Split 2: \n%s"%(str(y_NT[csj, :]))
        print

    else:
        assert zi_prop == zj_prop
        print "----MERGE----\n"
        [ci] = [x for x in clusters if i in x]
        [cj] = [x for x in clusters if j in x]
        [cm] = [x for x in clusters_prop if i in x]
        # print "i: %s\n"%(str(y_NT[i, :]))
        # print "j: %s\n"%(str(y_NT[j, :]))
        print "i :%d"%(i) 
        print "j :%d"%(j)

        print "Current 1: \n%s"%(str(y_NT[ci, :]))
        print "Current 2: \n%s"%(str(y_NT[cj, :]))
        print "Merge: \n%s"%(str(y_NT[cm, :]))
        print

def inference_restricted_split_merge(y_NT, shape, scale, alpha, M=5, num_itns=250, init_z_N=None, true_z_N=None):
    N, T = y_NT.shape

    C = N # maximum number of clusters equals number of observations
    if init_z_N is None:
        z_N = arange(N) # initialize each observation in its own cluster
        N_C = ones(C, dtype=int) # number of observations in each cluster
        y_CT = y_NT.copy() # sum of observations in each cluster
    else:
        z_N = init_z_N.copy()
        N_C = bincount(z_N, minlength=C)
        y_CT = zeros((N,T), dtype=int)
        for c in set(z_N):
            y_CT[c, :] += y_NT[z_N==c, :].sum(axis=0)

    active_clusters = set(z_N)
    inactive_clusters = set(range(C)) - active_clusters

    for itn in xrange(num_itns):
        if ((itn%200==0 or itn==0) and (true_z_N is not None)):
            print 'VI: %f bits (%f bits max.)' % (vi(true_z_N, z_N), log2(N))

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
            if uniform() < 0.5:
                z_launch_N[k] = zi_launch
            else:
                z_launch_N[k] = zj_launch

        N_launch_C = bincount(z_launch_N, minlength=C)

        # assign zi_launch -> 0, zj_launch -> 1
        N_launch_2 = zeros(2, dtype=int)
        N_launch_2[0] = N_launch_C[zi_launch]
        N_launch_2[1] = N_launch_C[zj_launch]

        y_launch_2T = zeros((2, T), dtype=int)
        y_launch_2T[0, :] += y_NT[z_launch_N == zi_launch, :].sum(axis=0)
        y_launch_2T[1, :] += y_NT[z_launch_N == zj_launch, :].sum(axis=0)

        # perform M intermediate restricted Gibbs scans
        dist = zeros(2)
        for m in xrange(M):
            for k in S:
                yk_T = y_NT[k, :]
                zk_launch = z_launch_N[k]

                N_launch_2[int(zk_launch == zj_launch)] -= 1
                y_launch_2T[int(zk_launch == zj_launch), :] -= yk_T

                dist = log(N_launch_2)
                dist[0] += (gammaln(shape + yk_T + y_launch_2T[0, :]) + (shape + y_launch_2T[0, :])*log(1.0/scale + N_launch_2[0])).sum()
                dist[0] -= (gammaln(shape + y_launch_2T[0, :]) + (shape + yk_T + y_launch_2T[0, :])*log(1.0/scale + N_launch_2[0] + 1)).sum()
                dist[1] += (gammaln(shape + yk_T + y_launch_2T[1, :]) + (shape + y_launch_2T[1, :])*log(1.0/scale + N_launch_2[1])).sum()
                dist[1] -= (gammaln(shape + y_launch_2T[1, :]) + (shape + yk_T + y_launch_2T[1, :])*log(1.0/scale + N_launch_2[1] + 1)).sum()
                t = float(log_sample(dist))

                N_launch_2[t] += 1
                y_launch_2T[t, :] += yk_T
                z_launch_N[k] = zi_launch if t == 0 else zj_launch

        if zi == zj: # SPLIT
            # print "SPLIT"
            # perform one final Gibbs scan from launch state
            
            prod_prob = 0.0 # maintain product of probabilities
            for k in S:
                yk_T = y_NT[k, :]
                zk_launch = z_launch_N[k]
                N_launch_2[int(zk_launch == zj_launch)] -= 1
                y_launch_2T[int(zk_launch == zj_launch), :] -= yk_T

                dist = log(N_launch_2)
                dist[0] += (gammaln(shape + yk_T + y_launch_2T[0, :]) + (shape + y_launch_2T[0, :])*log(1.0/scale + N_launch_2[0])).sum()
                dist[0] -= (gammaln(shape + y_launch_2T[0, :]) + (shape + yk_T + y_launch_2T[0, :])*log(1.0/scale + N_launch_2[0] + 1)).sum()
                dist[1] += (gammaln(shape + yk_T + y_launch_2T[1, :]) + (shape + y_launch_2T[1, :])*log(1.0/scale + N_launch_2[1])).sum()
                dist[1] -= (gammaln(shape + y_launch_2T[1, :]) + (shape + yk_T + y_launch_2T[1, :])*log(1.0/scale + N_launch_2[1] + 1)).sum()
                norm_const = logsumexp(dist)

                t = float(log_sample(dist))
                N_launch_2[t] += 1
                y_launch_2T[t, :] += yk_T
                z_launch_N[k] = zi_launch if t == 0 else zj_launch
                prod_prob += dist[t] - norm_const

            # evaluate split

            N_zi = N_C[zi]
            y_zi_T = y_CT[zi, :]
            N_zi_split, N_zj_split = N_launch_2
            y_zi_split_T = y_launch_2T[0,:]
            y_zj_split_T = y_launch_2T[1,:]

            P = log(alpha) + gammaln(N_zi_split) + gammaln(N_zj_split) - gammaln(N_zi)
            Q = -prod_prob
            L = 0.0
            L += T*(shape*log(1.0/scale) - gammaln(shape))
            L += (gammaln(shape + y_zi_split_T) - (shape + y_zi_split_T)*log(1.0/scale + N_zi_split)).sum()
            L += (gammaln(shape + y_zj_split_T) - (shape + y_zj_split_T)*log(1.0/scale + N_zj_split)).sum()
            L += ((shape + y_zi_T)*log(1.0/scale + N_zi) - gammaln(shape + y_zi_T)).sum()
            
            # print "SPLIT"
            # print_proposal(i, j, z_N, z_launch_N, y_NT)
            # print L

            acc = min([0, P + Q + L])
            if uniform() < exp(acc):

                z_N = z_launch_N.copy()
                active_clusters.add(zi_launch)

                y_CT[zj, :] = y_launch_2T[1, :]
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
            prod_prob = 0.0
            for k in S:
                yk_T = y_NT[k, :]
                zk = z_N[k]
                zk_launch = z_launch_N[k]
                N_launch_2[int(zk_launch == zj_launch)] -= 1
                y_launch_2T[int(zk_launch == zj_launch), :] -= yk_T

                dist = log(N_launch_2)
                dist[0] += (gammaln(shape + yk_T + y_launch_2T[0, :]) + (shape + y_launch_2T[0, :])*log(1.0/scale + N_launch_2[0])).sum()
                dist[0] -= (gammaln(shape + y_launch_2T[0, :]) + (shape + yk_T + y_launch_2T[0, :])*log(1.0/scale + N_launch_2[0] + 1)).sum()
                dist[1] += (gammaln(shape + yk_T + y_launch_2T[1, :]) + (shape + y_launch_2T[1, :])*log(1.0/scale + N_launch_2[1])).sum()
                dist[1] -= (gammaln(shape + y_launch_2T[1, :]) + (shape + yk_T + y_launch_2T[1, :])*log(1.0/scale + N_launch_2[1] + 1)).sum()
                norm_const = logsumexp(dist)

                t = int(zk == zj_launch)
                z_launch_N[k] = zk
                N_launch_2[t] += 1
                y_launch_2T[t] += yk_T

                prod_prob += dist[t] - norm_const
            
            # evaluate merge

            N_merge_C = bincount(z_merge_N, minlength=C)
            N_zi_merge = N_merge_C[zi_merge]
            N_zi = N_C[zi]
            N_zj = N_C[zj]
            y_zi_T = y_CT[zi, :]
            y_zj_T = y_CT[zj, :]
            y_zi_merge_T = y_NT[z_merge_N == zi_merge, :].sum(axis=0)

            P = -log(alpha) + gammaln(N_zi_merge) - gammaln(N_zi) - gammaln(N_zj)
            Q = prod_prob
            L = 0.0
            L += T*(gammaln(shape) - shape*log(1.0/scale))
            L += (gammaln(shape + y_zi_merge_T) - (shape + y_zi_merge_T)*log(1.0/scale + N_zi_merge)).sum()
            L += ((shape + y_zi_T)*log(1.0/scale + N_zi) - gammaln(shape + y_zi_T)).sum()
            L += ((shape + y_zj_T)*log(1.0/scale + N_zj) - gammaln(shape + y_zj_T)).sum()
            
            # print "MERGE"
            # print_proposal(i, j, z_N, z_merge_N, y_NT)
            # print L
            
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

def main():
    N = 200
    T = 52
    shape, scale = 3, 10
    alpha = 2

    seed(10)
    y_NT, true_z_N, true_delta_NT = generate_test_data(N, T, shape, scale, alpha)

    # print get_clusters(true_z_N)
    # print [y_NT[c, :] for c in get_clusters(true_z_N)][0]

    init_z_N = arange(N)
    # init_z_N = zeros(N, dtype=int)

    print "----Restricted SM----"
    z_N = inference_restricted_split_merge(y_NT, shape, scale, alpha, M=5, num_itns=201, init_z_N=init_z_N, true_z_N=true_z_N)

if __name__ == "__main__":
    main()


