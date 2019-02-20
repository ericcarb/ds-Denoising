import os
import numpy as np

def select_random_uniform (parent_dir):
	'''
	Selects a subset of each noise_file/snr combination set (11,572 files for 
	each combination) with a random uniform distribution.
	Change upper to control the proportion of files used.
	Change total according to the total number of files you have.
	'''
	upper = 4 # 1/(upper + 1) is the proportion of files used
	total = 11572
	mask = np.random.random_integers(0, upper, total) # need to update total according to max number of files to iterate over
	to_use = []
	for noise_dir in os.listdir(parent_dir):
		for snr_dir in os.listdir(os.path.join(parent_dir, noise_dir)):
			for i, f in enumerate(os.listdir(os.path.join(parent_dir, noise_dir, snr_dir))):
				if not mask[i]:
					to_use.append(f)
	return to_use