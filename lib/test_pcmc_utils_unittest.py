import unittest
from pcmc_utils import *
import numpy as np
import random

class TestpcmcUtils(unittest.TestCase):
	
	def setUp(self):
		pass
	
	def test_neg_L(self):
		pass
		
	def test_comp_Error(self):
		pass
	
	def test_solve_pcmc(self):
		pass
		
	def test_solve_ctmc(self):
		n = random.randint(50,100)
		Q = np.reshape(np.random.rand(n*n),(n,n))
		for i in range(n):
			Q[i,i]=0
			Q[i,i]-=np.sum(Q[i,:])
		pi = solve_ctmc(Q)
		self.assertAlmostEqual(np.sum(pi),1)
		self.assertAlmostEqual(np.sum(np.abs(np.dot(pi,Q))),0)
		
	def test_comp_P(self):
		n = random.randint(5,10)
		z = np.random.rand(n)
		x = np.reshape(np.array([[a]*(n-1) for a in z]),(n*(n-1)))
		print x.shape
		P = comp_P(x)
		for i in range(n):
			for j in range(n):
				if i==j:
					self.assertEqual(P[i,j],0)
				else:
					self.assertEqual(P[i,j],z[i])
			
	def test_cons_pairs(self):
		pass
		
	def test_solve_CTMC(self):
		pass

if __name__ == '__main__':
	unittest.main()
