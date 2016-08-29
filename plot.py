import numpy as np
from matplotlib import pyplot as plt
from lib.pcmc_utils import solve_ctmc

def plot_heatmap():
	P = np.load('pcmc_params.npy')
	n=P.shape[0]

	C = np.zeros(P.shape)
	for i in range(n):
		for j in range(i):
			C[i,j]=P[i,j]+P[j,i]
			C[j,i]=C[i,j]
			P[i,j]/=C[i,j]
			P[j,i]/=C[i,j]
	
	for i in range(n):
		C[i,i]=np.max(C)/2
		P[i,i]=.5
		

	d = solve_pcmc(P)
	idx = np.argsort(d)[::-1]
	P = P[:,idx];P= P[idx,:]
	C =C [:,idx];C=C[idx,:]
	f,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
	cbar1=ax1.matshow(P,cmap=plt.cm.coolwarm)
	ax1.set_xticklabels(['','1','2','3','4','5','6'])
	ax1.set_yticklabels(['','1','2','3','4','5','6'])
	ax1.tick_params(axis=u'both', which=u'both',length=0)
	cbar2=ax2.matshow(C,cmap=plt.cm.PuBuGn)
	ax2.set_xticklabels(['','1','2','3','4','5','6'])
	ax2.set_yticklabels(['','1','2','3','4','5','6'])	
	ax2.tick_params(axis=u'both', which=u'both',length=0)	
	plt.tight_layout(w_pad=0.2,h_pad=0)	
	cbar1 = plt.colorbar(cbar1)
	cbar2 = plt.colorbar(cbar2)
	cbar1.set_label('$p_{ij}/(p_{ij}+p_{ji})$')
	cbar2.set_label('$p_{ij}+p_{ji}$')
	plt.show()
	#plt.savefig('colormaps.png')
	
def plot_errors(split=.25):
	
	#set up for pyplot
	plt.rcParams.update({'font.size': 16})
	plt.rc('text', usetex=True)	
	plt.rc('font', family='serif')
	params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
	plt.rcParams.update(params)
	
	#load errors
	pcmc = np.load('pcmc_errors.npy')
	mnl = np.load('mnl_errors.npy')
	mmnl = np.load('mmnl_errors.npy')
	print pcmc
	print mnl
	#plotting variables
	nep = pcmc.shape[1] #recover number of ways training data was split
	percent = 100*(1-split)/nep*np.array(range(1,nep+1))
	
	#make plots
	plt.plot(percent,np.mean(mnl,axis=0),marker='x',color='r',label='MNL')
	plt.plot(percent,np.mean(mmnl,axis=0),marker='x',color='g',label='MMNL')
	plt.plot(percent,np.mean(pcmc,axis=0),marker='x',color='b',label='PCMC')
	
	#make it pretty
	plt.title('SFwork')
	plt.legend(loc=4)
	plt.xlabel(r'Percent of data used for training',labelpad=.1)	
	plt.xticks(percent)
	plt.xlim((0,100*(1-split)))
	plt.ylabel(r'Error')
	plt.ylim(ymin=0,ymax=.3)
	plt.tight_layout(w_pad=0.4,h_pad=0)

	
	plt.show()
	#plt.savefig(os.path.join(os.getcwd(),'pictures','errorplot.png'))

if __name__ == '__main__':
	np.set_printoptions(precision=4,suppress=True)
	plot_errors()
	plot_heatmap()