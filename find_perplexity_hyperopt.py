'''
README

v0.1 of tsne_spectra.py
'''

import numpy as np
import math
import time
from sklearn.manifold import TSNE
from hyperopt import hp, fmin, tpe

# used as test data
from sklearn.datasets import load_digits
import librosa
import os
import fnmatch
import json
import tSNE_audio as util



'''
from another implementation (ignore these two functions)
=================================================================================
'''
def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P



'''
Implementation of Equations 1-4 from https://arxiv.org/pdf/1708.03229.pdf
=================================================================================
'''
def l2norm2 (X):
	return np.linalg.norm(X, ord=2)**2


def q (y, i, j, n):
	numerator = (1 + l2norm2(y[i] - y[j]) )**(-1)
	denominator = 0
	for s in range(n):
		for t in range(n):
			if t != s:
				denominator += (1 + l2norm2(y[s] - y[t]) )**(-1)
	return numerator / denominator


def p_i_given_j (X, i, j, j_var=None):
	if j_var == None:
		j_var = np.std(X)**2
	numerator = math.exp( -1*l2norm2(X[i] - X[j]) / (2*j_var) )
	denominator = 0
	for s in range(n):
		if s != j:
			denominator += math.exp( -1*l2norm2(X[s] - X[j]) / (2*j_var) )
	return numerator / denominator


def p (X, i, j, n, j_var=None):
	if j_var == None:
		j_var = np.std(X)**2
	return (p_i_given_j(X, i, j, j_var) + p_i_given_j(X, j, i, j_var) ) / (2*n)


def KL (X, y, n):
	j_var = np.std(X)**2
	Sum = 0
	for j in range(n):
		for i in range(n):
			if i != j:
				p_ij = p(X, i, j, n, j_var)
				q_ij = q(y, i, j, n)
				Sum += p_ij * math.log(p_ij / q_ij)
	return Sum


def S (perp, X=np.array([])):
	tsne = TSNE(perplexity=perp, random_state=42)
	t0 = time.perf_counter()
	y = fit_transform(np.copy(X))
	t1 = time.perf_counter()
	print('Last t-SNE took {} seconds'.format(t1-t0))
	n = X.shape[0]
	return 2*KL(X, y, n) + (math.log(n)*perp/n)


def S_cached_kl (perp, X=np.array([])):
	tsne = TSNE(perplexity=perp, random_state=42)
	t0 = time.perf_counter()
	tsne.fit(X)
	t1 = time.perf_counter()
	print('Last t-SNE took {} seconds'.format(t1-t0))
	n = X.shape[0]
	return 2*tsne.kl_divergence_ + (math.log(n)*perp/n)


'''
Use this function if importing this file
=================================================================================
'''
def find_perplexity_hyperopt (X=np.array([]), tsne_params={}, 
							  lower_bound=1, upper_bound=50, max_evals=100):
	# would be nice to support passing tsne params to its init
	hp_fn = lambda perp : S_cached_kl(perp, X)
	space = hp.choice('p', np.arange(lower_bound, upper_bound+1))
	p = fmin(fn=hp_fn, space=space, algo=tpe.suggest, max_evals=max_evals)['p']
	return p


def find_perp (lower_bound=1, upper_bound=50, max_evals=100):
	feature_vectors = util.analyze_directory('/Users/ericcarb/ds-denoise/recordings/')
	hp_fn = lambda perp : run_tSNE(feature_vectors, '/Users/ericcarb/ml4a-ofx/apps/AudioTSNEViewer/bin/data/audiotsne.json', 2, perp)
	space = hp.choice('p', np.arange(lower_bound, upper_bound+1))
	p = fmin(fn=hp_fn, space=space, algo=tpe.suggest, max_evals=max_evals)['p']
	return p



'''
Example Usage
=================================================================================
'''

def run_tSNE(feature_vectors, tsne_path, tsne_dimensions, tsne_perplexity=30):
	print('run_tSNE')
	t = TSNE(n_components=tsne_dimensions, learning_rate=10, n_iter=5000, perplexity=tsne_perplexity, verbose=0, angle=0.1, random_state=42)
	tsne = t.fit_transform([f["features"] for f in feature_vectors])
	data = []
	for i,f in enumerate(feature_vectors):
		point = [ float(tsne[i,k] - np.min(tsne[:,k]))/(np.max(tsne[:,k]) - np.min(tsne[:,k])) for k in range(tsne_dimensions) ]
		data.append({"path":os.path.abspath(f["file"]), "point":point})
	with open(tsne_path, 'w') as outfile:
		json.dump(data, outfile)
	return t.kl_divergence_



if __name__ == '__main__':
	digits = load_digits()
	X = digits.data
	X = X[:50,:]


	p = find_perp()
	print(p)


	'''optimal_perplexity = find_perplexity_hyperopt(X, {}, 1, 50, 100)
	print('The optimal perplexity is {}'.format(optimal_perplexity))

	print('-'*82)
	p = optimal_perplexity
	tsne = TSNE(perplexity=p, verbose=2)
	tsne.fit(X)

	print('-'*82)
	tsne = TSNE(perplexity=10, verbose=2)
	tsne.fit(X)'''
