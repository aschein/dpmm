from collections import Counter
from numpy import argsort, bincount, log, log2, ones, unique, where, zeros
from numpy.random import poisson
from numpy.random.mtrand import dirichlet
from scipy.special import gammaln

from kale.math_utils import log_sample, sample, vi


def generate_data(V, D, l, alpha, beta):
    """
    Generates a synthetic corpus of documents from a Dirichlet process
    mixture model with multinomial mixture components (topics). The
    mixture components are drawn from a symmetric Dirichlet prior.

    Arguments:

    V -- vocabulary size
    D -- number of documents
    l -- average document length
    alpha -- Dirichlet process concentration parameter
    beta -- concentration parameter for the Dirichlet prior
    """

    T = D # maximum number of topics

    z_D = zeros(D, dtype=int)
    phi_TV = zeros((T, V))
    N_DV = zeros((D, V), dtype=int)

    for d in xrange(D):

        # generate a topic for this document

        dist = bincount(z_D).astype(float)
        dist[0] = alpha
        t = sample(dist)
        t = len(dist) if t == 0 else t
        z_D[d] = t

        if t == len(dist):
            phi_TV[t - 1, :] = dirichlet(beta * ones(V) / V, 1)

        # generate tokens for this document

        for v in sample(phi_TV[t - 1, :], num_samples=poisson(l)):
            N_DV[d, v] += 1

    z_D = z_D - 1

    return z_D, phi_TV, N_DV


def inference_algorithm_3(N_DV, alpha, beta, num_itns=250, true_z_D=None):
    """
    Algorithm 3.
    """

    D, V = N_DV.shape

    T = D # maximum number of topics

    N_D = N_DV.sum(1) # document lengths

    N_TV = zeros((T, V), dtype=int)
    N_T = zeros(T, dtype=int)

    z_D = range(D) # intialize every document to its own topic

    active_topics = set(unique(z_D))
    inactive_topics = set(xrange(T)) - active_topics

    for d in xrange(D):
        N_TV[z_D[d], :] += N_DV[d, :]
        N_T[z_D[d]] += N_D[d]

    D_T = bincount(z_D, minlength=T)

    for itn in xrange(num_itns):
        for d in xrange(D):

            old_t = z_D[d]

            D_T[old_t] -= 1
            N_TV[old_t, :] -= N_DV[d, :]
            N_T[old_t] -= N_D[d]

            log_dist = log(D_T)

            idx = old_t if D_T[old_t] == 0 else inactive_topics.pop()
            active_topics.add(idx)
            log_dist[idx] = log(alpha)

            for t in active_topics:
                log_dist[t] += gammaln(N_T[t] + beta)
                log_dist[t] -= gammaln(N_T[t] + N_D[d] + beta)
                tmp = N_TV[t, :] + beta / V
                log_dist[t] += gammaln(tmp + N_DV[d, :]).sum()
                log_dist[t] -= gammaln(tmp).sum()

            [t] = log_sample(log_dist)

            D_T[t] += 1
            N_TV[t, :] += N_DV[d, :]
            N_T[t] += N_D[d]

            z_D[d] = t

            if t != idx:
                active_topics.remove(idx)
                inactive_topics.add(idx)

        if true_z_D is not None:
            print 'VI: %f bits (%f bits max.)' % (vi(true_z_D, z_D), log2(D))

        for t in active_topics:
            print D_T[t], (N_TV[t, :] + beta / V) / (N_TV[t, :].sum() + beta)

        print len(active_topics)

    return z_D


def inference_algorithm_8(N_DV, alpha, beta, num_itns=250, true_z_D=None):
    """
    Algorithm 8.
    """

    M = 10

    D, V = N_DV.shape

    T = D + M - 1 # maximum number of topics

    N_D = N_DV.sum(1) # document lengths

    N_TV = zeros((T, V), dtype=int)
    N_T = zeros(T, dtype=int)

    z_D = range(D) # intialize every document to its own topic

    phi_TV = zeros((T, V))

    active_topics = set(unique(z_D))
    inactive_topics = set(xrange(T)) - active_topics

    for d in xrange(D):
        N_TV[z_D[d], :] += N_DV[d, :]
        N_T[z_D[d]] += N_D[d]

    D_T = bincount(z_D, minlength=T)

    for itn in xrange(num_itns):

        for t in active_topics:
            phi_TV[t, :] = dirichlet(N_TV[t, :] + beta / V, 1)

        for d in xrange(D):

            old_t = z_D[d]

            D_T[old_t] -= 1
            N_TV[old_t, :] -= N_DV[d, :]
            N_T[old_t] -= N_D[d]

            log_dist = log(D_T)

            idx = -1 * ones(M, dtype=int)
            idx[0] = old_t if D_T[old_t] == 0 else inactive_topics.pop()
            for m in xrange(1, M):
                idx[m] = inactive_topics.pop()
            active_topics |= set(idx)
            log_dist[idx] = log(alpha) - log(M)

            if idx[0] == old_t:
                phi_TV[idx[1:], :] = dirichlet(beta * ones(V) / V, M - 1)
            else:
                phi_TV[idx, :] = dirichlet(beta * ones(V) / V, M)

            for t in active_topics:
                log_dist[t] += (N_DV[d, :] * log(phi_TV[t, :])).sum()

            [t] = log_sample(log_dist)

            D_T[t] += 1
            N_TV[t, :] += N_DV[d, :]
            N_T[t] += N_D[d]

            z_D[d] = t

            idx = set(idx)
            idx.discard(t)
            active_topics -= idx
            inactive_topics |= idx

        if true_z_D is not None:
            print 'VI: %f bits (%f bits max.)' % (vi(true_z_D, z_D), log2(D))

        for t in active_topics:
            print D_T[t], (N_TV[t, :] + beta / V) / (N_TV[t, :].sum() + beta)

        print len(active_topics)

    return z_D


if __name__ == '__main__':

    V = 5
    D = 1000
    l = 1000
    alpha = 1.0
    beta = 0.1 * V

    z_D, phi_TV, N_DV = generate_data(V, D, l, alpha, beta)

    for t in argsort(bincount(z_D))[::-1]:
        idx, = where(z_D[:] == t)
        print len(idx), phi_TV[t, :]

    inf_z_D = inference_algorithm_3(N_DV, alpha, beta, 250, z_D)

    print 'VI: %f bits (%f bits max.)' % (vi(z_D, inf_z_D), log2(D))

    inf_z_D = inference_algorithm_8(N_DV, alpha, beta, 250, z_D)
