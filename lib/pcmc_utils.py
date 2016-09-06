import numpy as np
from scipy.optimize import minimize

def neg_L(x,C):
	"""negative log-likilhood of parameters P of a pcmc model given data in C
	
	Arguments:
	x- parameters for PCMC model
	C- dictionary containing choice sets and counts"""
	Q = comp_Q(x)
	L = 0
	for S in C:
		pi_S = np.log(solve_ctmc(Q[S,:][:,S]))
		for i in range(len(S)):
			L-=C[S][i]*pi_S[i]
	return L
	
def comp_error(x,C):
	"""computes expected L1 distance between probability vectors from a pcmc
	model and empirical distributions
	
	Arguments:
	x- model parameters
	C- empirical data
	"""
	err=0
	nsamp = np.sum(map(np.sum,C.values()))
	Q = comp_Q(x)		
	for S in C:
		pi=solve_ctmc(Q[S,:][:,S])
		err+=(np.sum(C[S])/nsamp)*np.sum(np.abs(pi-C[S]/np.sum(C[S])))
	return err	

def solve_ctmc(Q):
	"""Solves the stationary distribution of the CTMC whose rate matrix matches
	the input on off-diagonal entries. 
	Arguments:
	Q- rate matrix
	"""
	A=np.copy(Q)
	for i in range(Q.shape[0]):
		A[i,i] = -np.sum(Q[i,:])
	n=Q.shape[0]
	A[:,-1]=np.ones(n)
	b= np.zeros(n)
	b[n-1] = 1
	if np.linalg.matrix_rank(A)<Q.shape[0]:
		print Q
		print A
	return np.linalg.solve(A.T,b)

def comp_Q(x):
	"""reshapes PCMC parameter vector into rate matrix
	
	input:
	x- parameter vector of off-diagnoal entries of Q
	"""
	n = int(1+np.sqrt(4*len(x)+1))/2
	Q = np.empty((n,n))
	for i in range(n):
		row = np.insert(x[i*(n-1):(i+1)*(n-1)],i,0)
		Q[i,:]=row
	return Q

def cons_pairs(n):
	"""
	computes pairs of indices in flattened PCMC parameters x which correspond to
	q_ij and q_ji
	
	Arguments:
	n- number of elements in PCMC model 
	"""
	f = lambda i: (n-1)*(i%(n-1)+1)+i/(n-1)
	pairs = []
	for i in range(n):
		for a in [i*(n-1)+x for x in range(i,n-1)]:	
			pairs.append((a,f(a)))
	
	return pairs	


def infer(C,n,x=None,delta=1.0,maxiter=25):
	"""infers the parameters of a PCMC model using scipy.minimize to do MLE
	
	Arguments:
	C- training data
	n- number of elements in universe
	x- starting parameters
	delta- parameter of constraint q_ij+q_ji>=delta
	maxiter- number of iterations allowed to optimizer 
	"""
	bounds=[(10**(-9),None)]*(n*(n-1))
	if x is None:
		x = np.random.rand(n*(n-1))+delta/2.0
	cons=[]
	for (a,b) in cons_pairs(n):	
		cons.append({'type':'ineq','fun': lambda x,a=a,b=b:x[a]+x[b]-delta})
	res = minimize(neg_L,x,args=(C),bounds = bounds,constraints=tuple(cons),options={'disp':False,'maxiter':maxiter})
	return res.x
