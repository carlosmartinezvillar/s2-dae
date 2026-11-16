import os
import csv
import json
import argparse

################################################################################
# FUNCTIONS/HELPERS
################################################################################
def load_params(path):
	'''
	JSON Lines hyperparameter file -> {id: row}.
	'''
	with open(path,'r') as fp:
		return {row['id']:row for row in map(json.loads,filter(str.strip,fp))}


def best_probe_epoch(tsv_path,metric='iou_1',warmup=10):
	'''
	Best validation epoch of one probe log (probe_XXX.tsv), picked the same way
	train_condition.py picks the saved checkpoint: highest metric after warmup.
	train_base.py uses the same rule, so this also reads base_XXX.tsv.
	Returns that epoch's row as {column: value}, or None if no epoch passed warmup.
	'''
	with open(tsv_path,'r') as fp:
		rows = [r for r in csv.DictReader(fp,delimiter='\t') if int(r['epoch']) >= warmup]
	if not rows:
		return None

	best = max(rows,key=lambda r: float(r[metric])) # FIRST MAX ON TIES, LIKE THE STRICT > IN train_probe()
	return {k:(int(v) if k == 'epoch' else float(v)) for k,v in best.items()}


################################################################################
# BT PROBE
################################################################################
def best_bt_encoders(probe_dir,params_file='./hpo/bt_pretrain.json',metric='iou_1',top=5):
	'''
	Best BT encoder per encoder type ('cnn', 'vit') by the probe's validation metric.
	probe_dir holds the bt_probe logs (probe_XXX.tsv, model_XXX.done) copied from
	/log_pvc/bt_probe. A run counts only once its .done marker exists, so a sweep
	still in progress is reported as such instead of ranked as if complete.
	Prints the top runs per type and returns {encoder: best row}, where each row is
	the params row plus its best probe epoch under 'probe'.
	'''
	params  = load_params(params_file)
	ranked  = {}
	missing = []

	for i,hp in sorted(params.items()):
		tsv  = f'{probe_dir}/probe_{i:03}.tsv'
		done = os.path.isfile(f'{probe_dir}/model_{i:03}.done')
		best = best_probe_epoch(tsv,metric) if (done and os.path.isfile(tsv)) else None
		if best is None:
			missing.append(i)
			continue
		ranked.setdefault(hp.get('encoder','cnn'),[]).append({**hp,'probe':best})

	for runs in ranked.values():
		runs.sort(key=lambda r: r['probe'][metric],reverse=True)

	# REPORT
	if missing:
		print(f'{len(missing)} of {len(params)} probe runs not finished: {missing}')
	for encoder,runs in sorted(ranked.items()):
		print(f'--- {encoder.upper()}: top {min(top,len(runs))} of {len(runs)} runs by {metric} ---')
		for r in runs[:top]:
			print(f"id {r['id']:03} | {metric} {r['probe'][metric]:.5f} @ epoch {r['probe']['epoch']:03} | "
				f"proj_dim {r['proj_dim']} | init_chan {r['init_chan']} | vit_layers {r.get('vit_layers',1 if encoder == 'vit' else 0)} | "
				f"lrate {r['lrate']:g} | decay {r['decay']:g} | lambda_d {r['lambda_d']:g} | bt_lambda {r['bt_lambda']:g}")

	return {encoder:runs[0] for encoder,runs in ranked.items()}


################################################################################
# SUPERVISED BASELINE
################################################################################
def best_base(base_dir,params_file='./hpo/base_train.json',metric='iou_1',top=None):
	'''
	Supervised baseline runs (train_base.py) sorted by their best validation metric after
	warmup, i.e. the epoch of the saved base_XXX_e*.pth.tar checkpoint. base_dir holds the
	base_train logs (base_XXX.tsv, model_XXX.done) copied from /log_pvc/base_train; a run
	counts only once its .done marker exists. Prints the top runs (all if top is None) and
	returns the sorted list of params rows, each with its best epoch under 'base'.
	'''
	params  = load_params(params_file)
	ranked  = []
	missing = []

	for i,hp in sorted(params.items()):
		tsv  = f'{base_dir}/base_{i:03}.tsv'
		done = os.path.isfile(f'{base_dir}/model_{i:03}.done')
		best = best_probe_epoch(tsv,metric) if (done and os.path.isfile(tsv)) else None
		if best is None:
			missing.append(i)
			continue
		ranked.append({**hp,'base':best})

	ranked.sort(key=lambda r: r['base'][metric],reverse=True)

	# REPORT
	if missing:
		print(f'{len(missing)} of {len(params)} base runs not finished: {missing}')
	n = len(ranked) if top is None else min(top,len(ranked))
	print(f'--- BASE: top {n} of {len(ranked)} runs by {metric} ---')
	for rank,r in enumerate(ranked[:n],start=1):
		print(f"{rank:3} | id {r['id']:03} | {metric} {r['base'][metric]:.5f} @ epoch {r['base']['epoch']:03} | "
			f"vloss {r['base']['vloss']:.5f} | {r['encoder']}-{r['decoder']} | init_chan {r['init_chan']} | "
			f"vit_layers {r['vit_layers']} | batch {r['batch']} | lrate {r['lrate']:g} | decay {r['decay']:g}")

	return ranked


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--probe-dir',default=None,help='Directory of bt_probe logs (probe_XXX.tsv, model_XXX.done).')
	parser.add_argument('--params',default='./hpo/bt_pretrain.json',help='BT pretraining params file the probe ran on.')
	parser.add_argument('--base-dir',default=None,help='Directory of base_train logs (base_XXX.tsv, model_XXX.done).')
	parser.add_argument('--base-params',default='./hpo/base_train.json',help='Params file the base_train run used.')
	parser.add_argument('--top',type=int,default=None,help='Runs to print (probe: per encoder type, default 5; base: default all).')
	args = parser.parse_args()
	assert args.probe_dir or args.base_dir, "Give --probe-dir and/or --base-dir"

	if args.probe_dir:
		best = best_bt_encoders(args.probe_dir,args.params,top=5 if args.top is None else args.top)
		for encoder,row in sorted(best.items()):
			print(f"Best {encoder.upper()} encoder: id {row['id']:03}")

	if args.base_dir:
		best_base(args.base_dir,args.base_params,top=args.top)
