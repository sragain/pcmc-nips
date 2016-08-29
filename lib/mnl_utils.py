import numpy as np
from scipy.optimize import minimize
import random
from pcmc_utils import solve_ctmc

def ILSR(C,n):
	"""performs the ILSR algorithm to learn optimal MNL weights for choice data
	
	Arguments:
	C- dictionary containing choice data
	n- number of elements in the union of the choice sets
	epsilon- hyperparameter used for termination
	"""
	pi = np.ones(n).astype(float)/n
	diff = 1
	epsilon = 10**(-6)	
	while diff>epsilon:
		pi_ = pi
		lam = np.ones((n,n))*epsilon #initialization>0 prevents numerical issue
		for S in C:
			gamma = np.sum([pi[x] for x in S])
			pairs = [(i,j) for i in range(len(S)) for j in range(len(S)) if j!=i]
			for i,j in pairs:
				lam[S[j],S[i]]+=C[S][i]/gamma
			
		pi = solve_ctmc(lam)
		diff = np.linalg.norm(pi_-pi)
	return pi


def comp_error(x,C):
	err=0
	nsamp = np.sum(map(np.sum,C.values()))
	for S in C:
		pi_S = [x[i] for i in S]
		err+=(np.sum(C[S])/nsamp)*np.sum(np.abs(pi_S/np.sum(pi_S)-C[S]/np.sum(C[S])))
	return err	

