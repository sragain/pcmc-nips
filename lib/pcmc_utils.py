import numpy as np
from scipy.optimize import minimize

f=lambda :np.set_printoptions(suppress=True,precision=3)

def neg_L(x,C,cons):
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
	"""computes the expected L1 norm of the probability distribition implied by
	a PCMC model on P and the empirical distribution of a subset drawn from C
	
	Arguments:
	P- parameters for a PCMC model
	C- dictionary containing choice sets and counts
	"""
	err=0
	nsamp = np.sum(map(np.sum,C.values()))
	Q = comp_Q(x)		
	for S in C:
		pi=solve_ctmc(Q[S,:][:,S])
		err+=(np.sum(C[S])/nsamp)*np.sum(np.abs(pi-C[S]/np.sum(C[S])))
	return err	

def solve_ctmc(Q):
	"""gives the probability distribution implied by a pcmc model
	
	Arguments:
	Q- model parameters
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
	"""computes matrix of parameters Q from a flattened array of parameters for
	the pcmc model
	
	Arguments:
	x- n^2-n length array whose i-th n-1 entries are q_ij for j=/=i
	"""
	n = int(1+np.sqrt(4*len(x)+1))/2
	Q = np.empty((n,n))
	for i in range(n):
		row = np.insert(x[i*(n-1):(i+1)*(n-1)],i,0)
		Q[i,:]=row
	return Q

def cons_pairs(n):
	"""
	returns a list containing the pairs of indices of the
	flattened array of pcmc indices that correspond to 
	p_ij and p_ji
	"""
	f = lambda i: (n-1)*(i%(n-1)+1)+i/(n-1)
	pairs = []
	for i in range(n):
		for a in [i*(n-1)+x for x in range(i,n-1)]:	
			pairs.append((a,f(a)))
	
	return pairs	

def unit(n,a,b):
	y = np.zeros(n)
	y[a]=1;y[b]=1
	return y
	
def infer(C,n,x=None,delta=1.0,maxiter=25):
	"""write me"""
	bounds=[(10**(-9),None)]*(n*(n-1))
	if x is None:
		x = np.random.rand(n*(n-1))+delta/2.0
	cons=[]
	for (a,b) in cons_pairs(n):	
		cons.append({'type':'ineq','fun': lambda x,a=a,b=b:x[a]+x[b]-delta})
	res = minimize(neg_L,x,args=(C,cons),bounds = bounds,constraints=tuple(cons),options={'disp':False,'maxiter':maxiter})

	return res.x
