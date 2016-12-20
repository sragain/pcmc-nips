import numpy as np
from matplotlib import pyplot as plt
from lib.pcmc_utils import solve_ctmc

def plot(split,workFlag=True):
	"""generates heatmap and error plots from locally saved data
	to be run after infer.py
	
	Double commented lines 
	Aguments:
	split- proportion of data used for test set
	"""
	#set up for pyplot
	if workFlag:
		f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,5))
	else:
		f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,6))
	plt.rcParams.update({'font.size': 12})
	plt.rc('text', usetex=True)	
	plt.rc('font', family='serif')
	params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
	plt.rcParams.update(params)
	
	#load errors
	pcmc = np.load('pcmc_errors.npy')
	mnl = np.load('mnl_errors.npy')
	mmnl = np.load('mmnl_errors.npy')

	#plotting variables
	nep = pcmc.shape[1] #recover number of ways training data was split
	percent = 100*(1-split)/nep*np.array(range(1,nep+1))
	
	#make plots
	ax1.plot(percent,np.mean(mnl,axis=0),marker='o',color='r',label='MNL')
	ax1.plot(percent,np.mean(mmnl,axis=0),marker='o',color='g',label='MMNL')
	ax1.plot(percent,np.mean(pcmc,axis=0),marker='o',color='b',label='PCMC')
	
	#make it pretty
	ax1.legend(loc=3)
	ax1.set_xlabel(r'Percent of data used for training',labelpad=.1)	
	ax1.set_xticks(percent)
	ax1.set_xlim((0,80))
	ax1.set_ylabel(r'Error')
	ax1.set_ylim(ymin=0)

	
	#Now we plot the heatmap
	#load params
	Q= np.load('pcmc_params.npy')
	n=Q.shape[0]

	#sort the rows/columns by selection probabilities on whole set
	d = solve_ctmc(Q)
	idx = np.argsort(d)[::-1]
	Q = Q[:,idx];Q = Q[idx,:]

	#compute matrix C with c_ij=q_ij+q_ji (total rate for pairs)
	#also scale P so that p_ij+p_ji=1
	P = np.copy(Q)
	C = np.zeros(P.shape)
	C = C[:,idx];C = C[idx,:]		
	for i in range(n):
		for j in range(i):
			C[i,j]=P[i,j]+P[j,i]
			C[j,i]=C[i,j]
			P[i,j]/=C[i,j]
			P[j,i]/=C[i,j]
	
	#set meaningless diagonal values so that they will blend with colormap
	for i in range(n):
		C[i,i]=np.max(C)/2
		P[i,i]=.5
		
	

	
	#plot
	cbar1=ax2.matshow(P,cmap=plt.cm.coolwarm)
	if workFlag:
		ax2.set_xticklabels(['','1','2','3','4','5','6'])#
		ax2.set_yticklabels(['','1','2','3','4','5','6'])#
	else:
		ax2.set_xticklabels(['','1','2','3','4','5','6','7','8'])
		ax2.set_yticklabels(['','1','2','3','4','5','6','7','8'])	
	ax2.tick_params(axis=u'both', which=u'both',length=0)
	ax2.xaxis.tick_bottom()
	
	cbar2=ax3.matshow(C,cmap=plt.cm.PuBuGn)
	if workFlag:
		ax3.set_xticklabels(['','1','2','3','4','5','6'])#
		ax3.set_yticklabels(['','1','2','3','4','5','6'])#	
	else:
		ax3.set_xticklabels(['','1','2','3','4','5','6','7','8'])
		ax3.set_yticklabels(['','1','2','3','4','5','6','7','8'])		
	ax3.tick_params(axis=u'both', which=u'both',length=0)	
	ax3.xaxis.tick_bottom()
	
	cbar1 = plt.colorbar(cbar1,ax=ax2)
	cbar2 = plt.colorbar(cbar2,ax=ax3)
	cbar1.set_label('$q_{ij}/(q_{ij}+q_{ji})$')
	cbar2.set_label('$q_{ij}+q_{ji}$')
	if workFlag:
		ax2.set_title('SFwork')	
	else:
		ax2.set_title('SFshop')
	f.tight_layout(w_pad=0.2,h_pad=0)
	
	if workFlag:
		plt.savefig('workplot.png')
	else:
		plt.savefig('shopplot.png')

if __name__ == '__main__':
	split=.25;#update this if you change it in infer.py
	workFlag=True;#change to False to plot SFshop data
	plot(split=split,workFlag=workFlag)
