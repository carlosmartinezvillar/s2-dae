import itertools
import json
import os
import random
import numpy as np

################################################################################
# HYPERPARAMETER SEARCH
################################################################################
def search_lr_and_decay():
	'''
	Saves to 'hpo_1.json'.
	Search the hyperparameter space of learning rate and weight decay for
	S2SegDiff.
	'''

	# RANDOMIZE?
	# n_trials = 20
	# lrate = 10**np.random.uniform(np.log10(3e-5),-3,size=n_trials)
	# decay = np.concatenate([[0.0],10**np.random.uniform(-4,-1,size=n_trials-1)])

	# GRID
	learning_rates = [3e-5,5e-5,1e-4,2e-4,4e-4,8e-4] # ~x2 steps over [3e-5,1e-3]
	decays         = [0.0,1e-4,1e-3,1e-2,5e-2]      # 0 + log-spaced up to 5e-2

	# Define search space
	combinations = list(itertools.product(learning_rates,decays))

	# Define rows
	rows = []
	for i,(lr,wd) in enumerate(combinations):
		sample = {
			'id':i,
			'model':"S2SegDiff",
			'seed':476,
			'epochs':50,
			'bands':4,
			'labels':2,
			'lrate':lr,
			'decay':wd,
			'batch':32,
			'vit_layers':1,
			'mlp_ratio':4,
			'cnn_layers':2,
			'channels':32
		}

		rows.append(sample)

	# Save to JSON
	write_hp_file("hpo_1",rows)


################################################################################
# HELPER/OTHER FUNCTIONS
################################################################################
def write_hp_file(name,rows):
	# WRITE JSON FILE -- NEXT TO THIS SCRIPT, ONE JSON OBJECT PER LINE
	out_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"{name}.json")
	with open(out_file_path,'w') as fp:
		for line in rows[0:-1]:
			json.dump(line,fp)
			fp.write('\n')
		json.dump(rows[-1],fp)
	print(f"Parameter file written to {out_file_path} ({len(rows)} rows)")


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	set_seed(476)
	search_lr_and_decay()
