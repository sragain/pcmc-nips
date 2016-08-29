import unittest
from mmnl_utils import *
import numpy as np
import random

class TestmmnlUtils(unittest.TestCase):
	
	def setUp(self):
		pass
	
	def test_compute_probs(self):
		alpha = np.array([.5,1,2])
		gamma = np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,.5,.5]]).T
		target = np.array([.5,1.0,1.0,1.0])/3.5
		self.assertAlmostEqual(np.sum(np.abs(target-compute_probs(alpha,gamma))),0)
	
	def test_neg_L(self):
		pass
		
	def test_comp_Error(self):
		C = {(0,1):np.array([3.0,2.0]),(0,1,2):np.array([1.0,0.0,1.0])}
		alpha = np.array([.75,.25])
		gamma = np.array([[1.0,2.0,3.0],[1.0,0,1.0]]).T
		#this gives a distribution of .25,.25,.5 on the whole set and
		#.5 .5 on (0,1)
		x = np.array([.75,.25,1.0,1.0,2.0,0,3.0,1.0])
		self.assertAlmostEqual(comp_error(x,C,3),2.0/7)
	
	def test_infer(self):
		pass
		
if __name__ == '__main__':
	unittest.main()
