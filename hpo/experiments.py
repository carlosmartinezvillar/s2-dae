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
			'vit_layers':2,
			'mlp_ratio':5,
			'cnn_layers':3,
			'channels':32
		}

		rows.append(sample)

	# Save to JSON
	write_hp_file("hpo_1",rows)


def sample(spec):
    '''
    ('log',lo,hi)   -> log-uniform in [lo,hi]
    ('choice',[..]) -> uniform over the list
    '''
    if spec[0] == 'log':
        _, lo, hi = spec
        return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
    return spec[1][np.random.randint(len(spec[1]))]


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)


def set_experiments_unguided_diff(n=30,epochs=500,seed=476,path='./hpo/unguided.json')
	set_seed(seed)
	return


def set_experiments_bt(n=30, epochs=100, seed=476, path='./hpo/bt_encoder.json'):
    '''
    n random optimizer/loss samples for each architecture (proj_dim x init_chan), fixed batch.
    lrate = base_lrate * sqrt(batch/128), so base_lrate stays comparable if batch is changed later.
    '''
    set_seed(seed)

    space = {
        'base_lrate': ('log', 1e-4, 2e-3),       # lr at batch 256 -- multply by two
        'decay':      ('log', 1e-4, 5e-2),
        'lambda_d':   ('log', 1.0, 40.0),        # bt_lambda * proj_dim (off-diag terms grow ~D^2)
    }


    # FIXED
    batch = 128
    cnn_layers = 2
    encoder_type = 'cnn'


    # ARCHITECTURES (representation dim = 32*init_chan)
    grid = []
    for proj_dim in [1024, 2048, 4096]:
        for init_chan in [32, 48]:
            grid.append((proj_dim, init_chan))

    i = 0
    with open(path, 'w') as fp:
        for proj_dim, init_chan in grid:
            for _ in range(n):

                exp = {k: sample(v) for k, v in space.items()}
                exp['proj_dim']   = proj_dim
                exp['init_chan']  = init_chan
                exp['cnn_layers'] = cnn_layers
                exp['batch']      = batch  # images per batch
                exp['lrate']      = exp['base_lrate'] * (batch / 128) ** 0.5 #lrate adjust rule
                exp['encoder']    = encoder_type

                # ROUND (3 SIGNIFICANT FIGURES)
                for k in ('base_lrate', 'lrate', 'decay', 'lambda_d'):
                    exp[k] = float(f"{exp[k]:.3g}")
                exp['bt_lambda'] = float(f"{exp['lambda_d'] / proj_dim:.3g}")

                exp.update({'id': i, 'seed': seed, 'epochs': epochs, 'bands': 10})
                fp.write(json.dumps(exp) + '\n')
                i += 1


def set_experiments_train(n=30,epochs=500,seed=476,path='./hpo/unguided.json')
	set_seed(seed)
	return


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	set_seed(476)
	search_lr_and_decay()
