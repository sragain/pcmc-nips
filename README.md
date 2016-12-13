# pcmc-nips -- Pairwise Choice Markov Chains, NIPS 2016

This repository hosts data and code released in conjunction with the following paper:

- S. Ragain, J. Ugander (2016) "Pairwise Choice Markov Chains", NIPS.

The paper applies the PCMC model to two datasets: SFwork and SFshop. To reproduce the plot for SFwork seen in the paper, simply run infer.py and then plot.py. To average over more or fewer shuffles of the data, or to see more or fewer splits of the training data, or change the dataset, simply edit the top-level script environment at the bottom of infer.py. As written, the plotter plot.py must be updated if there are changes in train/test split.

The code can be run on the SFshop data by commenting in/out 2 lines in the script environment of infer.py, and changing a boolean in the script environment of plot.py.

### Detailed breakdown of files:

- SFwork.csv, SFshop.csv: CSV files containing the SFwork and SFshop datasets. Each row contains a choice, choice set sample. The index of the choice is followed by 0/1 entries marking whether the choice indexed by that column was in the choice set. The header gives the transportation options represented by each column. 

- worklist.p, shoplist.p: A pickled list of choice data comprising the SFwork and SFshop data respectively. The unpickled files are a list of choice-set and selection tuples of the form (S,idx) where S[idx] was chosen from S. The indexing matches the csv files. 

- infer.py: Shuffles and splits the data into training sets and a test set, then infers parameters for MNL, MMNL, and PCMC models
and outputs their inferential errors in the form of numpy arrays saved to the local directory. 

- plot.py: Reads in and plots inferential errors prepared by infer.py. 

See the following reference for more information:

[1] F. S Koppelman and C. Bhat. A self instructing course in mode choice modeling: multinomial and nested logit models. US Department of Transportation, Federal Transit Administration, 31, 2006.

Any comments, questions, or concerns should be directed to sragain at stanford.edu.