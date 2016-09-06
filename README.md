# pcmc-nips
Released in conjunction with "Pairwise Choice Markov Chains" S.Ragain, J. Ugander, To appear at NIPS 2016

To reproduce the plots seen in the paper, simply call
python infer.py
python plot.py

To average over more or fewer shuffles of the data, or to see more or fewer splits of the training data, or change the dataset,
simply edit the top-level script environement at the bottom of infer.py. Users may want to edit plot.py to reflect these changes.

A more detailed breakdown of the files:
Files:
worklist.p,shoplist.p: A pickled list of choice data comprising the SFwork and SFshop data respectively.
The unpickled files are a list of choice-set and selection tuples of the form (S,idx) where S[idx] was chosen from S .
For the work data:
0:'drive alone'	1:'Shared ride (2)'	2:'Shared ride (3+)'	3:'Transit'	4:'Bike'	5:'Walk',
and for the shop data:
0:'Transit'	1:'Shared ride (2)'	2:'Shared ride (3+)'	3:'Shared ride (2+) and drive alone'	4:'Shared ride (2/3+)'	
5:'Bike'	6:'Walk'	7:'drive alone'
See the following reference for more information:

[1] F. S Koppelman and C. Bhat. A self instructing course in mode choice modeling: multinomial and nested
logit models. US Department of Transportation, Federal Transit Administration, 31, 2006.

infer.py: Shuffles and splits the data into training sets and a test set, then infers parameters for MNL, MMNL, and PCMC models
and outputs their inferential errors in the form of numpy arrays saved to the local directory. 
plot.py: Reads in and plots inferential errors prepared by infer.py. 
