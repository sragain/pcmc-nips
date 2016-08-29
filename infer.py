import numpy as np
import matplotlib.pyplot as plt
from random import random
import lib.mnl_utils,lib.mmnl_utils,lib.pcmc_utils
import pickle
import os,sys

def split_samples(samples,nep,split=.25,epsilon=10**(-6)):
	"""splits a list of samples into nep dictionaries containing their summary
	statistics
	
	Arguments:
	samples- list of (Set, choice) tuples
	nep- number of ways to split input data
	epsilon- counts are initialized to epsilon rather than 0 for numerics
	"""
	Ctest = {}
	splitidx = int((1-split)*len(samples))
	testsamples = samples[splitidx:]
	for (S,choice) in testsamples:
		if S not in Ctest:
			Ctest[S]=np.ones(len(S))*epsilon
		Ctest[S][choice]+=1
	
	
	trainsamples = samples[:splitidx]
	trainlist = [{} for i in range(nep)]
	a = len(trainsamples)/nep
	for i in range(nep):
		for (S,choice) in trainsamples[i*a:(i+1)*a]:
			if S not in trainlist[i]:
				trainlist[i][S]=np.ones(len(S))*epsilon
			trainlist[i][S][choice]+=1
	return trainlist,Ctest
	
def run_sims(samples,n=6,nsim=10,nep=5,maxiter=25,split=.25):
	"""
	computes learning error on input models for input data for nsim simluations
	consisting of traning and computing test error on nep splits of the data
	
	Arguments:
	samples- list of samples
	n- number of choices in union of choice sets
	nsim- number of simulations to run
	nep- number of episodes per simulation
	split- proportion of samples used for testing
	"""
	mnl_errors=np.empty((nsim,nep))
	mmnl_errors=np.empty((nsim,nep))
	pcmc_errors=np.empty((nsim,nep))
	
	for sim in range(nsim):	
		print 'sim number %d' %(sim+1)
		np.random.shuffle(samples)
		
		#throw away any inferred parameters
		mnl_params = None;pcmc_params = None;mmnl_params = None
		
		#split data
		trainlist,Ctest = split_samples(samples,nep,split=split)
		Ctrain={}
		
		for ep in range(nep):
			#add new training data
			for S in trainlist[ep]:
				if S not in Ctrain:
					Ctrain[S]=trainlist[ep][S]
				else:
					Ctrain[S]+=trainlist[ep][S]		
			
			#infer parameters
			mnl_params = lib.mnl_utils.ILSR(C=Ctrain,n=n)								 
			mmnl_params = lib.mmnl_utils.infer(C=Ctrain,n=n,x=mmnl_params,maxiter=maxiter)
			pcmc_params = lib.pcmc_utils.infer(C=Ctrain,x=pcmc_params,n=n,maxiter=maxiter)		
			#track errors
			mnl_errors[sim,ep]=lib.mnl_utils.comp_error(x=mnl_params,C=Ctest)
			mmnl_errors[sim,ep]=lib.mmnl_utils.comp_error(x=mmnl_params,C=Ctest,n=n)
			pcmc_errors[sim,ep]=lib.pcmc_utils.comp_error(x=pcmc_params,C=Ctest)
			                                              
			
	np.save('mnl_errors.npy',mnl_errors)
	np.save('mmnl_errors.npy',mmnl_errors)
	np.save('pcmc_errors.npy',pcmc_errors)
	
if __name__=='__main__':

	
	#np.set_printoptions(suppress=True,precision=3)
	nsim=1000;nep=15;n=6;samples =pickle.load(open('worklist.p','rb')) 
	run_sims(samples=samples,n=n,nsim=nsim,nep=nep)
		
