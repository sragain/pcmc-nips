import numpy as np
from scipy.optimize import minimize

def compute_probs(alpha,gamma):
	"""returns the selection probabilities of an MMNL model
	
	Arguments:
	gamma - i-th column are parameters to i-th mnl
	alpha - alpha[i] is weight of i-th mnl
	
	Returns:
	vector of choice probabiles 
	"""
	K = len(alpha)
	n = gamma.shape[0]
	p = np.zeros(n)
	#normalize
	alpha/=np.sum(alpha)
	gamma/=np.sum(gamma,axis=0)
	return np.sum(np.multiply(alpha,gamma),axis=1)
	
def neg_L(x,K,C):
	"""negative log likelihood of an MMNL model
	
	Arguments:
	x- model parameters
	K- number of MNLs mixed
	C- data model evaluated against
	"""
	n = len(x)/K-1
	alpha = x[:K]
	gamma = np.reshape(x[K:],(n,K))

	L = 0
	for S in C:
		p = np.log(compute_probs(alpha,gamma[S,:]))
		for i in range(len(S)):
			L-=C[S][i]*p[i]
	return L

def comp_error(x,C,n,*args,**kwargs):
	""" computes the expected L1 norm between the empirical probabilities of a 
	choice set and those inferred by the MMNL with specified parameters
	
	Arguments:
	x- MMNL parameters
	K- number of MNL mixed
	C- data for evaluation
	"""
	err=0
	nsamp = np.sum(map(np.sum,C.values())).astype(float)

	K = len(x)/(n+1)
	alpha = x[:K]
	gamma = np.reshape(x[K:],(n,K))
	for S in C:
		p=compute_probs(alpha,gamma[S,:])
		err+=(np.sum(C[S])/nsamp)*np.sum(np.abs(p-C[S]/np.sum(C[S])))
	return err		

def comp_error_multi_init(x,C,n,*args,**kwargs):
	""" computes the expected L1 norm between the empirical probabilities of a 
	choice set and those inferred by the MMNL with specified parameters
	
	Arguments:
	x- tuple of list of MMNL parameters and best parameters
	K- number of MNL mixed
	C- data for evaluation
	"""
	err=0
	nsamp = np.sum(map(np.sum,C.values())).astype(float)
	x=x[1] #x is a tuple (all params,best params), we only compute error on best
	
	#unpack MMNL 
	K = len(x)/(n+1)
	alpha = x[:K]
	gamma = np.reshape(x[K:],(n,K))
	
	for S in C:
		p=compute_probs(alpha,gamma[S,:])#inferred by mmnl
		err+=(np.sum(C[S])/nsamp)*np.sum(np.abs(p-C[S]/np.sum(C[S])))
	return err		
	
def infer_multiple_init(C,n,x=None,maxiter=500,epsilon=10**(-6),K=5,ninit=5):
	"""returns the most likely of multiple trained mmnl models on training data
	
	
	Arguments:
	C- choice data
	n- size of union of choice sets
	x- list of ninit sets of mmnl parameters
	maxiter- maximum number of iterations allowed to scipy.minimize
	epsilon- used for parameter bounds to prevent numerical issues
	K- number of mnl mixed in each mmnl
	
	Returns:
	x- list of trained mmnl parameters
	x[idx]- most likely of the trained mmnl
	"""
	
	if x is None:
		x = [np.random.rand((n+1)*K)*(1-2*epsilon)+epsilon for i in range(ninit)]
	else:
		x=x[0]
		
	K = len(x[0])/(n+1)
	bounds = [(epsilon, None)]*(len(x[0]))
	best=0;idx=0
	for i in range(ninit):
		res = minimize (fun=neg_L,x0=x[i],args=(K,C),bounds = bounds,
						options={'disp':False,'maxiter':maxiter})
		x[i]=res.x
		if res.fun<best:
			best = res.fun
			idx=i
	return x,x[idx]
			
def infer(C,n,x=None,maxiter=500,epsilon=10**(-6),K=None):
	"""infers the parameters of a MMNL model on input data using MLE
	
	Arguments:
	C- choice data
	n- size of union of choice sets
	x- parameters to initialize optimization from
	maxiter- maximum number of iterations allowed to scipy.minimize
	epsilon- used for parameter bounds to prevent numerical issues
	"""
	
	if x is None:
		if K is None:
			K=n*(n-1)/(n+1)
		x = np.random.rand((n+1)*K)*(1-2*epsilon)+epsilon
	K = len(x)/(n+1)
	bounds = [(epsilon, None)]*(len(x))
	res = minimize(neg_L,x,args=(K,C),bounds = bounds,options={'disp':False,'maxiter':maxiter})
	return res.x
